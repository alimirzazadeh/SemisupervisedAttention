import os
from shutil import copy2, copytree, ignore_patterns
import datetime

def createExpFolderandCodeList(save_path,files=[]):
    #result folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    x = datetime.datetime.now()
    checkpointFolder = x.strftime("%x").replace('/','_') + "_" + x.strftime("%X").replace(':','_')
    save_path = os.path.join(save_path, checkpointFolder)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    code_folder_path = os.path.join(save_path, 'code')
    # if not os.path.exists(code_folder_path):
    #     os.makedirs(code_folder_path)

    img_folder_path = os.path.join(save_path, 'img')
    # if not os.path.exists(img_folder_path):
    #     os.makedirs(img_folder_path)

    copytree('./saved_figs/', img_folder_path)

    checkpoint_folder_path = os.path.join(save_path, 'checkpoints')

    copytree('./saved_checkpoints/', checkpoint_folder_path)


    copytree('./', code_folder_path, ignore=ignore_patterns('libs','data','data/*', 'saved*','README.md','imagenet_class_index.json','__pycache__','project_saver.py'))
    # #save code files
    # for file_name in os.listdir() + files:
    #     if file_name == 'metrics':
    #         copytree('./%s' % file_name, os.path.join(code_folder_path, 'train')) 
    #     if not os.path.isdir(file_name):
    #         copy2('./%s' % file_name, os.path.join(save_path, 'code', file_name))
createExpFolderandCodeList('saved_iterations')