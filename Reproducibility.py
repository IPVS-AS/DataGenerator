import argparse
import glob
import os
import shutil
import sys


def main(mode: str):

    #Clean build folder
    files = glob.glob('dist/*')
    for f in files:
        os.remove(f)

    if os.path.isdir('dist'): os.rmdir('dist')

    #create package
    os.system('python Setup.py install')

    os.system('python Setup.py bdist_wheel')

    whl_files = glob.glob('dist/*.whl')
    assert len(whl_files) > 0, 'No whl file was generated. Check Setup.py'

    #install package
    os.system(f'pip install {whl_files[0]}')

    #execute evaluation if desired 
    if mode == 'eval':
        os.system('python Evaluation.py')
    
    #Ensure that the folder for plots exists
    if not os.path.isdir('generated_plots'): 
        os.mkdir('generated_plots')

    #generate plots    
    os.system('python Generate_Plots.py')

    #copy to LaTex folder if exists
    source_dir = 'generated_plots'
    target_dir = 'DataGenerator/Figures/Evaluation'
    if os.path.isdir(target_dir):
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices=['plot', 'eval'], help='Determines whether the entire evaluation is performed or only the diagrams are created', required=True)
    args = vars(parser.parse_args(sys.argv[1:]))

    main(args['mode'])