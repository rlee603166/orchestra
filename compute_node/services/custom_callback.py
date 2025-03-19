from transformers.trainer_callback import TrainerCallback
import time
import json
import os


class CustomCallback(TrainerCallback):
    def __init__(self, log_file="training_data.json", stop_event=None):
        self.log_file = log_file
        self.stop_event = stop_event
        
        self.data = {
            "iterations": [],
            "loss": [],
            "learning_rate": [],
            "epoch": [],
        }
        
        self.log_data = {
            "steps": [], 
            "loss": [], 
            "accuracy": []
        }
        
        self.current_stats = {
            "current_loss": None,
            "current_lr": None,
            "current_epoch": None,
            "step": 0,
            "is_training": False,
            "last_updated": time.time()
        }
        
        self.current_step = 0
        self.last_loss = 0.0
        self.last_accuracy = 0.0

        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                json.dump(self.data, f)
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        self.current_stats["is_training"] = True
        return control
        
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        self.current_stats["is_training"] = False
        return control
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of an epoch"""
        pass
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of an epoch"""
        pass
        
    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of a training step"""
        if self.stop_event and self.stop_event.is_set():
            control.should_training_stop = True
            print("Stop signal detected in on_step_begin\nStopping...")
            return control
        return control

    def _save_current_stats(self, state):
        self.current_stats["current_loss"] = state.log_history[-1].get("loss") if state.log_history else None
        self.current_stats["current_lr"] = state.log_history[-1].get("learning_rate") if state.log_history else None
        self.current_stats["current_epoch"] = state.epoch
        self.current_stats["step"] = state.global_step
        self.current_stats["is_training"] = True
        self.current_stats["last_updated"] = time.time()
        
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of a training step"""
        self._save_current_stats(state)
        
        self.current_step = state.global_step
        
        if self.stop_event and self.stop_event.is_set():
            control.should_training_stop = True
            print("Training stop detected in on_step_end, stopping training...")
            return control
        
        if state.log_history and "loss" in state.log_history[-1]:
            loss = state.log_history[-1]["loss"]
            self.last_loss = loss
            
            self.data["iterations"].append(state.global_step)
            self.data["loss"].append(loss)
            self.data["learning_rate"].append(state.log_history[-1]["learning_rate"])
            self.data["epoch"].append(state.epoch)
            
            self.log_data["steps"].append(state.global_step)
            self.log_data["loss"].append(loss)
            
            with open(self.log_file, "w") as f:
                json.dump(self.data, f)
        
        return control
        
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
            
            if step not in self.log_data["steps"]:
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
            "accuracy": self.last_accuracy,
            "current_epoch": self.current_stats["current_epoch"],
            "is_training": self.current_stats["is_training"],
            "last_updated": self.current_stats["last_updated"]
        }