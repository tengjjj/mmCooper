
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor,
                target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class PointPillarLoss(nn.Module):
    def __init__(self, args):
        super(PointPillarLoss, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0

        self.cls_weight = args['cls_weight']
        self.reg_coe = args['reg']
        self.loss_dict = {}
        self.transformer_flag = args['transformer_flag']

    def forward(self, output_dict, target_dict):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        rm = output_dict['rm']
        psm = output_dict['psm']
        targets = target_dict['targets']
        if self.transformer_flag:
            bbx_off = output_dict['bbx_off']
            bbx_score = output_dict['bbx_score']

        cls_preds = psm.permute(0, 2, 3, 1).contiguous()

        box_cls_labels = target_dict['pos_equal_one']
        box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()

        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), 2,
            dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(psm.shape[0], -1, 1)
        one_hot_targets = one_hot_targets[..., 1:]

        cls_loss_src = self.cls_loss_func(cls_preds,
                                          one_hot_targets,
                                          weights=cls_weights)
        cls_loss = cls_loss_src.sum() / psm.shape[0]
        conf_loss = cls_loss * self.cls_weight

        # regression
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), -1, 7)
        targets = targets.view(targets.size(0), -1, 7)
        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm,
                                                                 targets)
        loc_loss_src =\
            self.reg_loss_func(box_preds_sin,
                               reg_targets_sin,
                               weights=reg_weights)
        reg_loss = loc_loss_src.sum() / rm.shape[0]
        reg_loss *= self.reg_coe
        
        if self.transformer_flag:
            assert not torch.isnan(bbx_off).any()
            bbx_off = bbx_off.permute(0, 2, 3, 1).contiguous()
            bbx_off = bbx_off.view(bbx_off.size(0), -1, 7)
            targets = targets.view(targets.size(0), -1, 7)
            bbx_targets = targets.clone()
            modify_target = torch.zeros(bbx_targets.shape, device=bbx_off.device, dtype=torch.float64)
            for b in range(bbx_targets.shape[0]):
                
                modify_target[b, output_dict['bbx_id'][b].long(), :] = bbx_targets[b, output_dict['bbx_id'][b].long(), :]

            box_preds_sin_bbx, reg_targets_sin_bbx = self.add_sin_difference(bbx_off,
                                                                    modify_target)
            loc_loss_src_bbx =\
                self.reg_loss_func(box_preds_sin_bbx,
                                reg_targets_sin_bbx,
                                weights=reg_weights)
            bbx_loss_reg = loc_loss_src_bbx.sum() / bbx_off.shape[0]
            bbx_loss_reg *= 0.2
            
            cls_modify_target = torch.zeros_like(one_hot_targets)
            bbx_cls_target = one_hot_targets.clone()
            for b in range(one_hot_targets.shape[0]):
                cls_modify_target[b, output_dict['bbx_id'][b].long(), :] = bbx_cls_target[b, output_dict['bbx_id'][b].long(), :]
            
            cls_loss_src_bbx = self.cls_loss_func(bbx_score,
                                            cls_modify_target,
                                            weights=cls_weights)
            loss_cls = cls_loss_src_bbx.sum() / bbx_score.shape[0]
            bbx_loss_cls = loss_cls * self.cls_weight


            total_loss = reg_loss + conf_loss + bbx_loss_cls + bbx_loss_reg

            self.loss_dict.update({'total_loss': total_loss,
                                'reg_loss': reg_loss,
                                'conf_loss': conf_loss, 
                                'bbx_loss_reg': bbx_loss_reg,
                                'bbx_loss_cls': bbx_loss_cls})
        else:
            total_loss = reg_loss + conf_loss

            self.loss_dict.update({'total_loss': total_loss,
                               'reg_loss': reg_loss,
                               'conf_loss': conf_loss})        

        return total_loss

    def cls_loss_func(self, input: torch.Tensor,
                      target: torch.Tensor,
                      weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        assert not torch.isnan(boxes1).any()
        assert not torch.isnan(boxes2).any()
        return boxes1, boxes2


    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss']
        reg_loss = self.loss_dict['reg_loss']
        conf_loss = self.loss_dict['conf_loss']

        if self.transformer_flag:
            bbx_loss_reg = self.loss_dict['bbx_loss_reg']
            bbx_loss_cls = self.loss_dict['bbx_loss_cls']
            if pbar is None:
                print("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                    " || Loc Loss: %.4f || Bbx cls Loss: %.4f || Bbx reg Loss: %.4f" % (
                        epoch, batch_id + 1, batch_len,
                        total_loss.item(), conf_loss.item(), reg_loss.item(), bbx_loss_cls.item(), bbx_loss_reg.item()))
            else:
                pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                    " || Loc Loss: %.4f || Bbx cls Loss: %.4f || Bbx reg Loss: %.4f" % (
                        epoch, batch_id + 1, batch_len,
                        total_loss.item(), conf_loss.item(), reg_loss.item(), bbx_loss_cls.item(), bbx_loss_reg.item()))
            
            writer.add_scalar('Bbx_loss_reg', bbx_loss_reg.item(),
                          epoch*batch_len + batch_id)
            writer.add_scalar('Bbx_loss_cls', bbx_loss_cls.item(),
                          epoch*batch_len + batch_id)
        else:
            if pbar is None:
                print("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                    " || Loc Loss: %.4f" % (
                        epoch, batch_id + 1, batch_len,
                        total_loss.item(), conf_loss.item(), reg_loss.item()))
            else:
                pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                    " || Loc Loss: %.4f" % (
                        epoch, batch_id + 1, batch_len,
                        total_loss.item(), conf_loss.item(), reg_loss.item()))

            
        writer.add_scalar('Regression_loss', reg_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('Confidence_loss', conf_loss.item(),
                          epoch*batch_len + batch_id)