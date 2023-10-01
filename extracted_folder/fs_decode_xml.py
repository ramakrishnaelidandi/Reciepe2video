# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:24:28 2019

@author: fame
"""

import xml.etree.ElementTree as ET  
import os  

def decode_recipe(recipe_root, id_to_word, recipename):
    recipe_steps = recipe_root.findall('steps_updated')
    assert( recipe_steps != None )
    recipe_steps_all = recipe_steps[0].findall('steps_updated')
    assert( recipe_steps_all != None ) 
    for cat in recipe_steps_all:
        tokens = cat.text.split(' ')
        ste = ""
        for token in tokens:
            if token =='':
                continue
            tok = int(token)
            ste = ste + id_to_word[tok].decode('utf8')
            ste = ste + " "
        cat.text = ste

    recipe_steps = recipe_root.findall('steps')
    assert( recipe_steps != None ) 
    recipe_steps_all = recipe_steps[0].findall('step')
    assert( recipe_steps_all != None ) 
    for cat in recipe_steps_all:
        tokens = cat.text.split(' ')
        ste = ""
        for token in tokens:
            if token =='':
                continue
            tok = int(token)
            ste = ste + id_to_word[tok].decode('utf8')
            ste = ste + " "
        cat.text = ste  
    
    recipe_steps = recipe_root.findall('ingredients')
    assert( recipe_steps != None )
    recipe_steps_all = recipe_steps[0].findall('ingredient')
    assert( recipe_steps_all != None )
    for cat in recipe_steps_all:
        tokens = cat.text.split(' ')
        ste = ""
        for token in tokens:
            if token =='':
                continue
            tok = int(token)
            ste = ste + id_to_word[tok].decode('utf8')
            ste = ste + " "
        cat.text = ste  
        
    tree.write(recipename, encoding="UTF-8",xml_declaration=True)

 
if __name__ == '__main__':
    encoded_dataset_path = 'ALL_RECIPES_without_videos/'
    
    recipe_names_fid = open("ALL_RECIPES.txt", "r")
    dirnames = recipe_names_fid.readlines()
    recipe_names_fid.close()
    count = 0
    id_to_word = eval(open('id2word_tasty.txt','r').read())
    for i in range(0, len(dirnames)):
        dirnames[i] = dirnames[i].replace('\n','')
        rec_f = encoded_dataset_path + dirnames[i] + '/recipe_encoded.xml'
        tree = ET.parse(rec_f)
        cat_root = tree.getroot()
        
        curr_file_path =  encoded_dataset_path +  dirnames[i]
        if not os.path.exists(curr_file_path):
            os.makedirs(curr_file_path)
            
        decode_recipe(cat_root, id_to_word, curr_file_path +'/recipe.xml' )
        print(' [*] DONE ', i, ' ', dirnames[i])

 