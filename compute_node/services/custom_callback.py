from transformers.trainer_callback import TrainerCallback
import json

class CustomCallback(TrainerCallback):
    def __init__(self, log_file="training_data.json"):
        self.log_file = log_file
        self.log_data = {"steps": [], "loss": [], "accuracy": []}
        self.current_step = 0
        self.last_loss = 0.0
        self.last_accuracy = 0.0
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        pass
        
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        pass
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of an epoch"""
        pass
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of an epoch"""
        pass
        
    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of a training step"""
        pass
        
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of a training step"""
        pass
        
    def on_evaluate(self, args, state, control, **kwargs):
        """Called after evaluation"""
        pass
        
    def on_save(self, args, state, control, **kwargs):
        """Called when saving the model"""
        pass
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs"""
        if logs is None:
            return
        
        step = state.global_step
        loss = logs.get("loss", None)
        
        if "eval_accuracy" in logs:
            accuracy = logs.get("eval_accuracy", 0.0)
        else:
            accuracy = None

        if loss is not None:
            self.last_loss = loss
            self.current_step = step
            self.log_data["steps"].append(step)
            self.log_data["loss"].append(loss)
            
            if accuracy is not None:
                self.last_accuracy = accuracy
                self.log_data["accuracy"].append(accuracy)
            else:
                self.log_data["accuracy"].append(self.last_accuracy)
                
            self._save_training_data()
            
            print(f"Step: {step}, Loss: {loss}, Accuracy: {self.last_accuracy if accuracy is None else accuracy}")

    def _save_training_data(self):
        with open(self.log_file, "w") as f:
            json.dump(self.log_data, f)
            
    def get_current_stats(self):
        return {
            "step": self.current_step,
            "loss": self.last_loss,
            "accuracy": self.last_accuracy
        }
