from setting import setting
import subprocess
def makecommand(obj,c):
    if obj==[]:
        cmd=c+['--savefolder',"".join(c[2:]).replace('--','')]
        print('\033[32m'+f'{" ".join(c)}'+'\033[0m')
        subprocess.run(cmd)
        return 0
    arg,values=obj[0]
    if type(values)!=type([]):values=[values]
    for v in values:
        makecommand(obj[1:],c+[f'--{arg}', f'{v}'])
        
print('s',setting)
makecommand(setting,['python3','main.py'])
