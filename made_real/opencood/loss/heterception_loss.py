import torch
from torch import nn
import numpy as np
from opencood.loss.point_pillar_loss import PointPillarLoss
from icecream import ic 

class HeterceptionLoss(PointPillarLoss):
    def __init__(self, args):
        super(HeterceptionLoss, self).__init__(args)
        self.ssl = args['ssl']
        self.l2_loss = nn.MSELoss()
    
    def forward(self, output_dict, target_dict, suffix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        total_loss = super().forward(output_dict, target_dict, suffix)

        ###### self-supervise learning loss ######
        if f'poi_feature_pred{suffix}' in output_dict and output_dict[f'poi_feature_pred{suffix}'].shape[0]!=0:
            ssl_loss = self.l2_loss(output_dict[f'poi_feature_pred{suffix}'],
                                    output_dict[f'poi_feature_gt{suffix}'],)
            ssl_loss *= self.ssl['weight']
            total_loss += ssl_loss
            self.loss_dict.update({'total_loss': total_loss.item(),
                                    'ssl_loss': ssl_loss.item()})
        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer = None, suffix=""):
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
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        iou_loss = self.loss_dict.get('iou_loss', 0)
        ssl_loss = self.loss_dict.get('ssl_loss', 0)


        print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || SSL Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, dir_loss, iou_loss, ssl_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss', reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss', cls_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss', dir_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Iou_loss', iou_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('SSL_loss', ssl_loss,
                            epoch*batch_len + batch_id)