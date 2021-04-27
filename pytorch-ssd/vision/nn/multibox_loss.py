import torch.nn as nn
import torch.nn.functional as F
import torch
import math


from ..utils import box_utils


class MultiboxLoss(nn.Module):

    class DrLoss():

      def count_dr_loss_f(self, confidence, labels):

          neg_lambda = 0.1/math.log(3.5)
          pos_lambda = 1
          tau = 4
          L = 6
          margin = 0.5

          num_classes = confidence.shape[1]
          dtype = labels.dtype
          device = labels.device
          class_range = torch.arange(1, num_classes + 1, dtype=dtype, device=device).unsqueeze(0)
          t = labels.unsqueeze(1)
          pos_ind = (t == class_range)
          neg_ind = (t != class_range) * (t >= 0)
          pos_prob = confidence[pos_ind].sigmoid()
          neg_prob = confidence[neg_ind].sigmoid()
          neg_q = F.softmax(neg_prob/neg_lambda, dim=0)
          neg_dist = torch.sum(neg_q * neg_prob)
          if pos_prob.numel() > 0:
              pos_q = F.softmax(-pos_prob/pos_lambda, dim=0)
              pos_dist = torch.sum(pos_q * pos_prob)
              loss = tau*torch.log(1.+torch.exp(L*(neg_dist - pos_dist+margin)))/L
          else:
              loss = tau*torch.log(1.+torch.exp(L*(neg_dist - 1. + margin)))/L
          return loss

    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.

        Also implemented DR loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)
        self.drloss = self.DrLoss()

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute DR loss, classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
         
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
    
        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
        dr_loss = self.drloss.count_dr_loss_f(confidence.reshape(-1, num_classes), labels[mask])
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        ##print (f"DR Loss: {dr_loss.item():.4f}")

        return smooth_l1_loss/num_pos, dr_loss, classification_loss/num_pos