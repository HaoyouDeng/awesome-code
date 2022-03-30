from tensorboardX import SummaryWriter
import math

writer = SummaryWriter(log_dir='./log') # log_dir = './path/to/log'

for epoch in range(100):
    mAP = -math.log(epoch+1)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mAP2', mAP, epoch)

'''
tensorboard --logdir=./path/to/the/folder --port 8123   # 8123 is an example
use browser to visit localhost:8123/
'''
