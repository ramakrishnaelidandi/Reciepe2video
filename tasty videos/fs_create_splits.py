# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 01:11:36 2019

@author: fame
"""

import xml.etree.ElementTree as ET
import os
import json
import xml.etree.cElementTree as etree
import sys
import os.path


def read_ingredients(recipe_root, recipe_title):
    ingred_cats = recipe_root.findall('ingredients')
    assert( ingred_cats != None )

    all_ingreds = ingred_cats[0].findall('ingredient')
    assert( all_ingreds != None )

    ingredientC_list = []
    for ingC in all_ingreds:
        curr_ingC = ingC.text.lower()
        split_ings = curr_ingC.split(';')

        ind_ingredient = -1
        ind_amounts1 = -1
        ind_amounts2 = -1
        if len(split_ings)  == 1 :
            ind_ingredient = 0
        elif len(split_ings)  == 2 :
            ind_ingredient = 1
            ind_amounts1 = 0
        else:
            ind_ingredient = 1
            ind_amounts1 = 0
            ind_amounts2 = 2

        amounts = [] # not returned!
        if not ind_amounts1 == -1:
            amounts.append(split_ings[ind_amounts1].strip())
        if not ind_amounts2 == -1:
            amounts.append(split_ings[ind_amounts2].strip())

        curr_vals = split_ings[ind_ingredient].strip()
        curr_vals_split = curr_vals.split(',')

        curr_vals_act = curr_vals_split[0].strip()
        ingredientC_list.append(curr_vals_act)

    return ingredientC_list


def read_steps(recipe_root):
    recipe_steps = recipe_root.findall('steps_updated')
    assert( recipe_steps != None )

    recipe_steps_all = recipe_steps[0].findall('steps_updated')
    assert( recipe_steps_all != None )

    step_list = []
    for cat in recipe_steps_all:
        curr_step = cat.text.lower().strip()
        step_list.append(curr_step)
    return step_list


def read_label_frames(recipe_steps, dirnames_ii, annotation_steps):

    frame_labels = []
    for xi in range (0, len(recipe_steps) ):
        if annotation_steps[xi] == '-1,-1':
            curr_range = []
        else:
            annots = annotation_steps[xi].split(',')
            start_frame = int(annots[0] )
            end_frame = int(annots[1])
            if start_frame == 1:
                start_frame_actual = 1
            else:
                start_frame_actual = start_frame*5 - 4
            end_frame_actual = end_frame*5

            curr_range = (start_frame_actual, end_frame_actual)

        frame_labels.append(curr_range)
    return frame_labels


if __name__ == "__main__":
    pathG = os.path.dirname(sys.argv[0])

    split_path  = 'SPLITS/split_4022/'

    train_lines = [train_lines.rstrip('\n')       for train_lines       in open( split_path + 'TRAIN_4022.txt')]
    val_lines   = [test_others_lines.rstrip('\n') for test_others_lines in open( split_path + 'VAL_4022.txt') ]
    test_lines  = [test_zero_lines.rstrip('\n')   for test_zero_lines   in open( split_path + 'TEST_4022.txt')]

    recipe_path =  'ALL_RECIPES_without_videos/'
    all_recipes = [all_recipes.rstrip('\n')    for all_recipes  in open( 'ALL_RECIPES.txt')]

    for kki in range( len(all_recipes) ):

        src_xml   = recipe_path +   all_recipes[kki]   + '/recipe.xml'
        xmlDoc_f = open(src_xml, 'r')
        xmlDocData_f = xmlDoc_f.read()
        xmlDoc_f.close()
        xmlDocTree = etree.XML(xmlDocData_f)

        recipe_title = xmlDocTree.find('url').text
        recipe_title = recipe_title.replace('\n','')
        recipe_title = recipe_title.replace('https://tasty.co/recipe/', '')
        print(' [*] ', kki, ' ', all_recipes[kki])

        tree = ET.parse(src_xml)
        recipe_root = tree.getroot()

        recipe_ingredients = read_ingredients(recipe_root, recipe_title)
        recipe_steps = read_steps(recipe_root)

        src_annotation = recipe_path +   all_recipes[kki]   +  '/csvalignment.dat'
        annotation_steps =  [line.rstrip('\n') for line in open(src_annotation)]

        recipe_annotations = read_label_frames(recipe_steps, all_recipes[kki], annotation_steps )

        if all_recipes[kki]   in val_lines:
            split = 'val'
        elif  all_recipes[kki]  in test_lines :
            split = 'test'
        else:
            split = 'train'

        recipe_dict = {"title":  recipe_title,
                       "split": split,
                       "ingredients": recipe_ingredients,
                       "steps": recipe_steps,
                       "annotations": recipe_annotations  }
        with open('recipes_processed.txt', 'a') as json_file:
            json.dump(recipe_dict, json_file)

