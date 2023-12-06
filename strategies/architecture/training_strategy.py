

class Evaluation_Strategy:
    def train(self, model, training_data, test_data, optimizer, scheduler, pred_loss_func, pred_loss_goal, opt):
        return super().train()

    def train_epoch(self, model, training_data, optimizer, pred_loss_func, pred_loss_goal, opt):
        pass

    def eval_epoch(self, model, test_data, pred_loss_func, pred_loss_goal, opt):
        pass