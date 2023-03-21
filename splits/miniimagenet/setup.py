import tarfile

f_train = './train.tar'
f_val = './val.tar'
f_test = './test.tar'

def extract(fname):
    ap = tarfile.open(fname)
    ap.extractall('.')
    ap.close()

extract(f_train)
extract(f_val)
extract(f_test)