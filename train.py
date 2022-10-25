import wandb

from utils.setting import Setting
from utils.trainer import Trainer

if __name__ == '__main__':

    args, logger = Setting().run()

    trainer = Trainer(args, logger)

    if args.wandb:
        name = f'{args.method}'
        wandb.init(project='CS쉐어링 method 선택',
                   name=name,
                   config=vars(args))

    for epoch in range(args.epochs):
        logger.info(f'Start Training Epoch {epoch}')
        trainer.train_epoch(epoch)
        logger.info(f'Finish Training Epoch {epoch}')

    logger.info('Training Finished')
