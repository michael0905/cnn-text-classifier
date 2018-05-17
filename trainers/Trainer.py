from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np

class Trainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(Trainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        valid_loss, valid_acc = self.valid_step()

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'train_loss': loss,
            'train_acc': acc,
            'valid_loss': valid_loss,
            'valid_acc': valid_acc,
        }
        print(summaries_dict)
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.inputs_placeholder: batch_x,
                     self.model.labels_placeholder: batch_y,
                     self.model.dropout_placeholder: self.config.dropout_rate,
                     self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_op, self.model.loss, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc

    def valid_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.valid_size, is_training=False))
        feed_dict = {self.model.inputs_placeholder: batch_x,
                     self.model.labels_placeholder: batch_y,
                     self.model.dropout_placeholder: 1,
                     self.model.is_training: False}
        _, loss, acc = self.sess.run([self.model.train_op, self.model.loss, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc
