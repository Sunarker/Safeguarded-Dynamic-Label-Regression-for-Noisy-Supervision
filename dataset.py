# disturb the CIFAR-10 with the specific noise
import numpy as np
import sys
import struct 

# binary format
# <1 byte for fine label> <3072 byte for pixels>

# change to new binary format
# <4 bytes for indices> <1 byte for fine label> <3072 byte for pixels>

def gen_noise_block(NOISY_PROPORTION):
  # blocks
  T = np.eye(10)
  T[9][9], T[9][1] = 1 - NOISY_PROPORTION, NOISY_PROPORTION
  T[2][2], T[2][0] = 1 - NOISY_PROPORTION, NOISY_PROPORTION
  T[4][4], T[4][7] = 1 - NOISY_PROPORTION, NOISY_PROPORTION
  T[3][3], T[3][5] = 1 - NOISY_PROPORTION, NOISY_PROPORTION

  return T

def dis_noise(label,T):
   return xrange(10)[np.argmax(np.random.multinomial(size=1,n=1,pvals=T[label]))]

def index_and_inject_data(filename, start_count, NOISY_PROPORTION):
   with open(filename,'rb') as f:
      with open(filename[:-4] + '_noise_%.2f_with_index'% NOISY_PROPORTION + filename[-4:],'wb') as w:
        T = gen_noise_block(NOISY_PROPORTION)
        count = start_count
        e = f.read(3073)
        while e:
          label = ord(e[0])
          #print(label)
          dis_label = dis_noise(label,T)
          dis_e = struct.pack('I',count) + chr(dis_label) + e[1:]
          w.write(dis_e)
          count += 1
          e = f.read(3073)

def index_data(filename,start_count):
   with open(filename,'rb') as f:
      with open(filename[:-4] + '_with_index' + filename[-4:],'wb') as w:
        count = start_count
        e = f.read(3073)
        while e:
          ind_e = struct.pack('I',count) + e
          w.write(ind_e)
          count += 1
          e = f.read(3073)

if __name__ == '__main__':
   for NOISY_PROPORTION in [0, 0.1, 0.3, 0.5, 0.7, 0.9]:
     index_and_inject_data('data/cifar10/cifar-10-batches-bin/data_batch_1.bin',0,NOISY_PROPORTION)
     index_and_inject_data('data/cifar10/cifar-10-batches-bin/data_batch_2.bin',10000,NOISY_PROPORTION)
     index_and_inject_data('data/cifar10/cifar-10-batches-bin/data_batch_3.bin',20000,NOISY_PROPORTION)
     index_and_inject_data('data/cifar10/cifar-10-batches-bin/data_batch_4.bin',30000,NOISY_PROPORTION)
     index_and_inject_data('data/cifar10/cifar-10-batches-bin/data_batch_5.bin',40000,NOISY_PROPORTION)
   index_data('data/cifar10/cifar-10-batches-bin/test_batch.bin',0)

