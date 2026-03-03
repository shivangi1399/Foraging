import sys 
import os
import glob

if len(sys.argv) == 1:
    raise ValueError('missing pipe input')  
pipename = sys.argv[1]

print('No file name given, please enter manually and I will construct the filename.')
subj = input("Subject name:\n")
#print(f'You entered {subj}')
if subj[0].isnumeric():
    root = '/as/projects/OWzeronoise/*/%s'%(subj)
else:
    root = '/as/projects/MWzeronoise/*/%s'%(subj)
rec = input("Recording date:\n")
#print(f'You entered {rec}')

# get list of files that matches pattern
pattern = '%s/%s/*/*'%(root, rec) 
dirs = list(filter(os.path.isdir, glob.glob(pattern)))

for ii, this_dir in enumerate(dirs):
    print(ii, this_dir)
idx = input("Type the number of the archive folder to analyse:\n")

print('You selected: ', dirs[int(idx)])
ans = input("Is this correct? y/n:\n")
if ans != 'y':
    sys.exit("Wrong archive folder give, press ctrl c and try again")
archive_folder = dirs[int(idx)]

# write archive_folder to fifo pipe
with open(pipename, 'w') as fifo:
    print(archive_folder, file=fifo)

