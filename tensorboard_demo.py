from tensorboardX import SummaryWriter
import math

writer = SummaryWriter(log_dir='./log') # log_dir = './path/to/log'

x = []
for epoch in range(100):
    mAP = -math.log(epoch+1)

    x = [mAP, 2*mAP, -mAP]

    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mAP2', mAP, epoch)
    writer.add_scalars('mAP3', {'aa':x[0],'bb':x[1]}, epoch)
    print(epoch)

'''
tensorboard --logdir=./path/to/the/folder --port 8123   # 8123 is an example
use browser to visit localhost:8123/
'''
