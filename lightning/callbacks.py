from lightning.pytorch.callbacks import Callback
import asyncio
import telegram_send
from pytorch_lightning.utilities.exceptions import MisconfigurationException


async def send_message(message) -> None:
    """Send a message"""
    await telegram_send.send(messages=[message])

class TelegramCallback(Callback):
    """this requires you to set the telegram_send bot first
    - pip install telegram-send -> telegram-send --configure
    - methods to override can be found here https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.Callback.html
    """
    def on_train_start(self, trainer, pl_module):

        if not trainer.logger:
            raise MisconfigurationException(
                'Cannot use LearningRateMonitor callback with Trainer that has no logger.'
            )
        try:
            loop = asyncio.get_event_loop()
            opt_params=trainer.optimizers[0].param_groups[0]
            loop.run_until_complete(send_message(f"Training Started for {trainer.max_epochs-trainer.current_epoch} Epochs\n\n LR: {opt_params['lr']}"))
        except Exception as e:
            print(f"Failed to send message ",e)
            
    def on_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics.items()
        message=f"EPOCH {trainer.current_epoch} END\n"
        message += "\n".join([f"{k}:{v:.3f}" for k,v in m])
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(send_message(message))
        except Exception as e:
            print(f"Failed to send message ",e)

    def on_validation_end(self,trainer,pl_module):
        m = trainer.callback_metrics.items()
        message=f"VALIDATION END Epoch {trainer.current_epoch}\n\n"
        message += "\n".join([f"{k} : {v:.3f}" for k,v in m])
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(send_message(message))
        except Exception as e:
            print(f"Failed to send message ",e)

