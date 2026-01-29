import sys
import pywed
import pickle
nshot =int(sys.argv[1])

t_ignitron, _, _ = pywed.tsbase(nshot, 'RIGNITRON')
t_ignitron = t_ignitron.item()

sys.stdout.buffer.write(pickle.dumps(t_ignitron))