#!/usr/bin/env python
"""
    Training on a single process
"""
from __future__ import division
import numpy as np
import pandas as pd
import codecs
import os
import re
import argparse
from argparse import Namespace

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn import ensemble

from PIL import Image
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem, rdShapeHelpers
from rdkit.Chem.QED import qed
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from rdkit.Chem import MolStandardize
from rdkit.Chem import rdMolAlign
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit import RDConfig
import sascorer

import cairosvg
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from rdkit import rdBase, RDLogger
import warnings
warnings.filterwarnings('ignore')
rdBase.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.*')  # https://github.com/rdkit/rdkit/issues/2683

DrawingOptions.atomLabelFontSize = 55
DrawingOptions.dotsPerAngstrom = 100
DrawingOptions.bondLineWidth = 3.0

# Set up features to use in FeatureMap
fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
fdef = AllChem.BuildFeatureFactory(fdefName)

fmParams = {}
for k in fdef.GetFeatureFamilies():
    fparams = FeatMaps.FeatMapParams()
    fmParams[k] = fparams

keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable',
        'ZnBinder', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe')


def get_FeatureMapScore(query_mol, ref_mol):
    featLists = []
    for m in [query_mol, ref_mol]:
        rawFeats = fdef.GetFeaturesForMol(m)
        # filter that list down to only include the ones we're intereted in
        featLists.append([f for f in rawFeats if f.GetFamily() in keep])
    fms = [FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=fmParams) for x in featLists]
    fms[0].scoreMode = FeatMaps.FeatMapScoreMode.Best
    fm_score = fms[0].ScoreFeats(featLists[1]) / min(fms[0].GetNumFeatures(), len(featLists[1]))

    return fm_score


def calc_SC_RDKit_score(query_mol, ref_mol):
    fm_score = get_FeatureMapScore(query_mol, ref_mol)

    protrude_dist = rdShapeHelpers.ShapeProtrudeDist(query_mol, ref_mol,
                                                     allowReordering=False)
    SC_RDKit_score = 0.5 * fm_score + 0.5 * (1 - protrude_dist)

    return SC_RDKit_score


def calc_2D_similarity(s1, s2):
    mol1 = Chem.MolFromSmiles(s1)
    mol2 = Chem.MolFromSmiles(s2)
    if mol1 is None:
        return -1.0

    f1 = AllChem.GetMorganFingerprint(mol1, 2, useCounts=True, useFeatures=True)
    f2 = AllChem.GetMorganFingerprint(mol2, 2, useCounts=True, useFeatures=True)
    return DataStructs.TanimotoSimilarity(f1, f2)





# chooseone begin with 1,2,...
"""
datatype:if src then [1:] else [:]
chooseone:if the file contain not only onecase, you can choose one by order
repeatsize: if you use chooseone then you must use repeatsize

return: sequences list
"""
def read_seqs_from_file(path, datatype, src_type):
    assert datatype in ['src','target','sample']
    if datatype == 'src':
        if src_type == 'N':
            with codecs.open(path, 'r', 'utf-8') as f:
                seqs = [''.join(line.strip().split(' ')) for line in f.readlines()]
            return seqs
        else:
            with codecs.open(path, 'r', 'utf-8') as f:
                seqs = [''.join(line.strip().split(' ')[1:]) for line in f.readlines()]
            return seqs


    with codecs.open(path, 'r', 'utf-8') as f:
        seqs = [''.join(line.strip().split(' ')) for line in f.readlines()]

    return seqs



def remove_dummys(smi_string):
    try:
        smi = Chem.MolToSmiles(Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(smi_string), \
                                                                       Chem.MolFromSmiles('*'), \
                                                                       Chem.MolFromSmiles('[H]'), True)[0]))
    except:
        smi = ""

    return smi



# 输入应该是gen,及去掉了*的fregments smile形式
# 如果生成的分子包含标准的fregments,没有在上面添加，即match,返回true
# 否则，返回false
def juice_is_standard_contains_fregments(gen, frags):
    "input generated molecules and the starting fragments of original molecules \
     return to the generated linker and  the two linker sites in fragments"

    m = Chem.MolFromSmiles(gen)
    matches = m.GetSubstructMatches(Chem.MolFromSmiles(frags))
    if matches:

        atoms = m.GetNumAtoms()
        for index,match in enumerate(matches):
            atoms_list = list(range(atoms))
            for i in match:
                atoms_list.remove(i)

            # 目前atoms_list和linker_list是去掉match的原子了
            linker_list = atoms_list.copy()
            for i in atoms_list:
                atom = m.GetAtomWithIdx(i)
                for j in atom.GetNeighbors():
                    linker_list.append(j.GetIdx())
            linker_list = list(set(linker_list))
            # print(f'after linker_list: {linker_list}')
            sites = list(set(linker_list).difference(set(atoms_list)))  #找到和fregments连接的那两个原子（即*
            # print(f'after sites: {sites}')

            # 如果上面去掉fregment之后得到两部分linker那就说明要么这个match方式有问题
            # 要么fregment上面增生了原子
            if len(sites) == 2:
                return True

    else:
        return False

    return False

def juice_is_standard_contains_fregments2(gen, frags):
    "input generated molecules and the starting fragments of original molecules \
     return to the generated linker and  the two linker sites in fragments"

    m = Chem.MolFromSmiles(gen)
    matches = m.GetSubstructMatches(Chem.MolFromSmiles(frags))
    if matches:
        return True
    return False


def fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    size = 2048
    nfp = np.zeros((1, size), np.int32)
    for idx,v in fp.GetNonzeroElements().items():
        nidx = idx%size
        nfp[0, nidx] += int(v)
    return nfp


def calQED(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        score = qed(mol)
        return score
    else:
        return -1

def calMW(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        weight = Descriptors.MolWt(mol)
        return weight
    return -1

def calCLOGP(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        mol_ClogP = Chem.Crippen.MolLogP(mol)
        return mol_ClogP
    return -10

def calNOS(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        has_sulphur = [atom.GetAtomicNum()==16 for atom in mol.GetAtoms()]
        # print(f'src:{src,smile,has_sulphur.count(True)}')
        return float(has_sulphur.count(True))
    else:
        return -1

def calActive(smi, clf):
    # with open(clf_path, "rb") as f:
    #     clf = joblib.load(f)
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = fingerprints_from_mol(mol)
        score = clf.predict(fp)
        return float(score)
    else:
        return -1


def calSIM_3D(smi, ref):
    ref_mol = Chem.MolFromSmiles(ref)
    gen_mol = Chem.MolFromSmiles(smi)
    if gen_mol:
        try:
            ref_mol = Chem.AddHs(ref_mol)
            Chem.AllChem.EmbedMolecule(ref_mol, randomSeed=10)
            Chem.AllChem.UFFOptimizeMolecule(ref_mol)

            gen_mol = Chem.AddHs(gen_mol)
            Chem.AllChem.EmbedMolecule(gen_mol, randomSeed=10)
            Chem.AllChem.UFFOptimizeMolecule(gen_mol)

            pyO3A = rdMolAlign.GetO3A(gen_mol, ref_mol).Align()
            return calc_SC_RDKit_score(gen_mol, ref_mol)
        except Exception:
            return -0.5
    else:
        return -1.0

def calSA(smi, ref):
    # ref_mol = Chem.MolFromSmiles(ref)
    gen_mol = Chem.MolFromSmiles(smi)
    if gen_mol:
        try:
            sa_score = sascorer.calculateScore(gen_mol)
            return sa_score
        except Exception:
            return -0.5
    else:
        return -1.0

def calSIM_3D_seed(smi, ref,seed):
    ref_mol = Chem.MolFromSmiles(ref)
    gen_mol = Chem.MolFromSmiles(smi)
    if gen_mol:
        try:
            ref_mol = Chem.AddHs(ref_mol)
            Chem.AllChem.EmbedMolecule(ref_mol, randomSeed=seed)
            Chem.AllChem.UFFOptimizeMolecule(ref_mol)

            gen_mol = Chem.AddHs(gen_mol)
            Chem.AllChem.EmbedMolecule(gen_mol, randomSeed=seed)
            Chem.AllChem.UFFOptimizeMolecule(gen_mol)

            pyO3A = rdMolAlign.GetO3A(gen_mol, ref_mol).Align()
            return calc_SC_RDKit_score(gen_mol, ref_mol)
        except Exception:
            return -0.5
    else:
        return -1.0


"""
:return one smile's function score
"""
def calScore(smi, ref, functiontype, clf):
    assert functiontype in ['QED', 'MW', 'CLOGP', 'NOS', 'SIM_3D','SIM_3D1','SIM_3D2','linker_length','QED_SA','tanimoto','M_SIM_QED','activity_model']
    if functiontype == 'QED':
        return calQED(smi)
    if functiontype == 'MW':
        return calMW(smi)
    if functiontype == 'CLOGP':
        return calCLOGP(smi)
    if functiontype == 'NOS':
        return calNOS(smi)
    if functiontype == 'SIM_3D' or functiontype == 'SIM_3D1' or functiontype == 'SIM_3D2' :
        return calSIM_3D(smi, ref)
    if functiontype == 'QED_SA':
        return calSA(smi, ref)
    if functiontype == 'tanimoto':
        return calc_2D_similarity(smi, ref)
    if functiontype == 'M_SIM_QED':
        return calQED(smi)
    if functiontype == 'activity_model':
        return calActive(smi, clf)


"""
:return judge wheather one smile is containing fragments both sides
"""
def juiceIsValid(src, smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        # return True
        # juice wheather the gensmile contains the fregment(src) standardly
        # src去掉*
        src_new = remove_dummys(src)
        isstandard = juice_is_standard_contains_fregments2(smi, src_new)
        if isstandard:
            return True
        return False
    return False



# score,src,smi,isValid
"""
ziplst: score must at the first location 
i: the num of smiles you want to save by scores
savepath: the case save path
datatype: [prior, agent]
"""
def drawTopimol(ziplst, i, savepath, datatype):
    assert i <= len(ziplst)
#     print(i,len(ziplst))
    savenum = 0
    ziplst.sort(reverse=True)     # 默认按照第一个元素排序
    for score,score2d,src,smi,isValid in ziplst:
        mol = Chem.MolFromSmiles(smi)
        if not mol or not isValid:
            continue
        # Draw.MolToFile(mol,f'{savepath}/{datatype}_{savenum}_match_{round(score, 4)}_{round(score2d, 4)}_old.png')
        Draw.MolToFile(mol,f'{savepath}/{datatype}_{savenum}_match_{round(score, 4)}_{round(score2d, 4)}.svg')
        cairosvg.svg2png(url=f'{savepath}/{datatype}_{savenum}_match_{round(score, 4)}_{round(score2d, 4)}.svg', \
                         write_to=f'{savepath}/{datatype}_{savenum}_match_{round(score, 4)}_{round(score2d, 4)}.png')
        # print(f'src:{src}')
        # src = remove_dummys(src)
        # matches = mol.GetSubstructMatches(Chem.MolFromSmarts(src))
        # if isValid:
        #     # v = False if not mol else True
        #     # print(f'v:{v,smi}')
        #     # img = Draw.MolToImage(mol, size=(500, 500), highlightAtoms=matches[0])
        #     # img.save(os.path.join(f'{savepath}/{datatype}_{savenum}_match_{round(score, 4)}.png'))
        #     Draw.MolToFile(mol,f'{savepath}/{datatype}_{savenum}_match_{round(score, 4)}_{round(score2d, 4)}.png')
        # else:
        #     # img = Draw.MolToImage(mol, size=(500, 500))
        #     # img.save(os.path.join(f'{savepath}/{datatype}_{savenum}_notmatch_{round(score, 4)}.png'))
        #     Draw.MolToFile(mol,f'{savepath}/{datatype}_{savenum}_notmatch_{round(score, 4)}_{round(score2d, 4)}.png')
        savenum = savenum + 1
        if savenum >= i:
            break

def getXlabel(opt):
    # assert opt.score_function_type in ['QED', 'MW', 'CLOGP', 'NOS', 'SIM_3D', 'tanimoto']
    if opt.score_function_type == 'CLOGP':
        return "clogP"
    if opt.score_function_type == 'SIM_3D' or opt.score_function_type == 'SIM_3D1'\
         or opt.score_function_type == 'SIM_3D2' or opt.score_function_type == 'QED_SA':
        return "SC score"
    if opt.score_function_type == 'tanimoto':
        return "Tanimoto Similarity"
    if opt.score_function_type == 'activity_model':
        return "Activity Value"
    # if opt.score_function_type == 'M_SIM_QED':
    #     return
    return opt.score_function_type

def draw(prior_qedlst, agent_qedlst, savepath, opt):
    score_type = getXlabel(opt)
    sns.set_style("white")  # 风格选择包括："white", "dark", "whitegrid", "darkgrid", "ticks"

    plt.figure(figsize=(8, 4), dpi=500)
    sns.distplot(prior_qedlst, label='prior', color="#536B97",
                 hist_kws={'histtype': "stepfilled"}  # 默认为bar,可选barstacked,step,stepfilled
                 )
    sns.distplot(agent_qedlst, label='agent', color="#6AA781",
                 hist_kws={'histtype': "stepfilled"}  # 默认为bar,可选barstacked,step,stepfilled
                 )

    plt.ylabel(f'Density(n={opt.beam_size})', fontdict={'size': 9})
    plt.xlabel(f'{score_type}', fontdict={'size': 9})
    plt.title(f'The {score_type} distribution of prior/agent', fontdict={'size': 9})
    plt.legend()

    plt.savefig(os.path.join(f'{savepath}/distribution_{opt.id}_{score_type}_type1.jpg'))


def draw2(prior_qedlst, agent_qedlst, savepath, opt):
    score_type = getXlabel(opt)
    sns.set_style("white")  # 风格选择包括："white", "dark", "whitegrid", "darkgrid", "ticks"

    plt.figure(figsize=(8, 4), dpi=500)
    sns.kdeplot(prior_qedlst, shade=True, label='prior', color="#536B97")
    sns.kdeplot(agent_qedlst, shade=True, label='agent', color="#6AA781")

    plt.ylabel(f'Density(n={opt.beam_size})', fontdict={'size': 9})
    plt.xlabel(f'{score_type}', fontdict={'size': 9})
    plt.title(f'The {score_type} distribution of prior/agent', fontdict={'size': 9})
    plt.legend()
    plt.savefig(os.path.join(f'{savepath}/distribution_{opt.id}_{score_type}_type2.jpg'))


def drawMulti2D3D(tmpdf,savepath,opt,min=0):
    color_a = "#DD6F72"
    color_p = "#8E8E8E"
    tmpdf_prior = tmpdf.loc[tmpdf["prior_isValid"] == True]
    tmpdf_agent = tmpdf.loc[tmpdf["agent_isValid"] == True]
    bins_num = 20
    hist = True
    y_min, y_max = min, 1
    x_min, x_max = min, 1

    fig = plt.figure(constrained_layout=True, figsize=(9, 8),dpi=500)
    gs = fig.add_gridspec(10, 10)
    ax1 = fig.add_subplot(gs[2:, -1])

    sns.distplot(tmpdf_prior['prior_2dsim'], kde_kws=dict(linewidth=2.5),ax=ax1, vertical=True, hist_kws={'histtype': "stepfilled"}, color=color_p)
    sns.distplot(tmpdf_agent['agent_2dsim'], kde_kws=dict(linewidth=2.5),ax=ax1, vertical=True, hist_kws={'histtype': "stepfilled"}, color=color_a)

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylabel('')
    ax1.set_ylim([y_min, y_max])
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlabel('Density', size=12)


    ax2 = fig.add_subplot(gs[2:, :-1])
    ax2.scatter(tmpdf_prior['prior_qed'], tmpdf_prior['prior_2dsim'], s=120, marker='o', label='Prior', linewidths=2,
                alpha=0.7, color=color_p)
    ax2.scatter(tmpdf_agent['agent_qed'], tmpdf_agent['agent_2dsim'], s=120, marker='^', label='Agent', linewidths=2,
                alpha=0.7, color=color_a)
    # ax2.legend(prop={'size': 16}, markerscale=2, loc=(0.05, 0.8),fontsize=70)
    ax2.legend(loc='best',fontsize=20,frameon=False)
    ax2.tick_params(labelsize=20)
    ax2.set_ylim([y_min, y_max])
    ax2.set_xlim([x_min, x_max])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlabel('SCscore', size=22)
    ax2.set_ylabel('Tanimoto Similarity', size=22)

    ax3 = fig.add_subplot(gs[1, :-1])
    sns.distplot(tmpdf_prior['prior_qed'], kde_kws=dict(linewidth=2.5),ax=ax3, vertical=False, hist_kws={'histtype': "stepfilled"}, color=color_p)
    sns.distplot(tmpdf_agent['agent_qed'], kde_kws=dict(linewidth=2.5),ax=ax3, vertical=False, hist_kws={'histtype': "stepfilled"}, color=color_a)

    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlabel('')
    ax3.spines['top'].set_visible(False)
    ax3.set_ylabel('Density', size=12)

    plt.savefig(os.path.join(f'{savepath}/distribution_{opt.id}_Multi_2D3D_type1.jpg'))

def drawMulti2D3D_type1old(tmpdf,savepath,opt,min=0):
    color_a = "#DD6F72"
    color_p = "#8E8E8E"
    tmpdf_prior = tmpdf.loc[tmpdf["prior_isValid"] == True]
    tmpdf_agent = tmpdf.loc[tmpdf["agent_isValid"] == True]
    bins_num = 20
    hist = True
    y_min, y_max = min, 1
    x_min, x_max = min, 1

    fig = plt.figure(constrained_layout=True, figsize=(9, 8))
    gs = fig.add_gridspec(10, 10)
    ax1 = fig.add_subplot(gs[2:, -1])

    # sns.distplot(prior_qedlst, label='prior', color="#536B97",
    #              hist_kws={'histtype': "stepfilled"}  # 默认为bar,可选barstacked,step,stepfilled
    #              )
    sns.distplot(tmpdf_prior['prior_2dsim'], ax=ax1, vertical=True, hist_kws={'histtype': "stepfilled"}, color=color_p)
    sns.distplot(tmpdf_agent['agent_2dsim'], ax=ax1, vertical=True, hist_kws={'histtype': "stepfilled"}, color=color_a)
    # sns.distplot(tmpdf_prior['prior_2dsim'], ax=ax1, vertical=True, hist=hist, bins=bins_num, color=color_p)
    # sns.distplot(tmpdf_agent['agent_2dsim'], ax=ax1, vertical=True, hist=hist, bins=bins_num, color=color_a)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylabel('')
    ax1.set_ylim([y_min, y_max])
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2 = fig.add_subplot(gs[2:, :-1])

    ax2.scatter(tmpdf_prior['prior_qed'], tmpdf_prior['prior_2dsim'], marker='o', label='prior', linewidths=1.7,
                alpha=0.6, color=color_p)
    ax2.scatter(tmpdf_agent['agent_qed'], tmpdf_agent['agent_2dsim'], marker='^', label='agent', linewidths=1.7,
                alpha=0.6, color=color_a)

    ax2.legend(prop={'size': 16}, markerscale=2, loc=(0.05, 0.8))
    ax2.tick_params(labelsize=15)
    ax2.set_ylim([y_min, y_max])
    ax2.set_xlim([x_min, x_max])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlabel('SCscore', size=15)
    ax2.set_ylabel('Tanimoto Similarity', size=15)

    ax3 = fig.add_subplot(gs[1, :-1])
    sns.distplot(tmpdf_prior['prior_qed'], ax=ax3, vertical=False, hist_kws={'histtype': "stepfilled"}, color=color_p)
    sns.distplot(tmpdf_agent['agent_qed'], ax=ax3, vertical=False, hist_kws={'histtype': "stepfilled"}, color=color_a)
    # sns.distplot(tmpdf_prior['prior_qed'], ax=ax3, vertical=False, hist=hist, bins=bins_num, color=color_p)
    # sns.distplot(tmpdf_agent['agent_qed'], ax=ax3, vertical=False, hist=hist, bins=bins_num, color=color_a)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlabel('')
    ax3.spines['top'].set_visible(False)
    plt.savefig(os.path.join(f'{savepath}/distribution_{opt.id}_Multi_2D3D_type1.jpg'))

def drawMulti2D3D2(tmpdf,savepath,opt,min=0):
    color_a = "#DD6F72"
    color_p = "#8E8E8E"
    tmpdf_prior = tmpdf.loc[tmpdf["prior_isValid"] == True]
    tmpdf_agent = tmpdf.loc[tmpdf["agent_isValid"] == True]

    y_min, y_max = min, 1
    x_min, x_max = min, 1

    fig = plt.figure(constrained_layout=True, figsize=(9, 8))
    gs = fig.add_gridspec(10, 10)
    ax1 = fig.add_subplot(gs[2:, -1])

    sns.kdeplot(tmpdf_prior['prior_2dsim'], ax=ax1, vertical=True, shade=True, color=color_p)
    sns.kdeplot(tmpdf_agent['agent_2dsim'], ax=ax1, vertical=True, shade=True, color=color_a)

    # sns.distplot(tmpdf_prior['prior_2dsim'], ax=ax1, vertical=True, hist_kws={'histtype': "stepfilled"}, color=color_p)
    # sns.distplot(tmpdf_agent['agent_2dsim'], ax=ax1, vertical=True, hist_kws={'histtype': "stepfilled"}, color=color_a)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylabel('')
    ax1.set_ylim([y_min, y_max])
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2 = fig.add_subplot(gs[2:, :-1])
    ax2.scatter(tmpdf_prior['prior_qed'], tmpdf_prior['prior_2dsim'], marker='o', label='prior', linewidths=1.7,
                alpha=0.6, color=color_p)
    ax2.scatter(tmpdf_agent['agent_qed'], tmpdf_agent['agent_2dsim'], marker='^', label='agent', linewidths=1.7,
                alpha=0.6, color=color_a)


    ax2.legend(prop={'size': 16}, markerscale=2, loc=(0.05, 0.8))
    ax2.tick_params(labelsize=15)
    ax2.set_ylim([y_min, y_max])
    ax2.set_xlim([x_min, x_max])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlabel('SCscore', size=15)
    ax2.set_ylabel('Tanimoto Similarity', size=15)

    ax3 = fig.add_subplot(gs[1, :-1])
    sns.kdeplot(tmpdf_prior['prior_qed'], ax=ax3, vertical=False, shade=True, color=color_p)
    sns.kdeplot(tmpdf_agent['agent_qed'], ax=ax3, vertical=False, shade=True, color=color_a)
    # sns.distplot(tmpdf_prior['prior_qed'], ax=ax3, vertical=False, hist_kws={'histtype': "stepfilled"}, color=color_p)
    # sns.distplot(tmpdf_agent['agent_qed'], ax=ax3, vertical=False, hist_kws={'histtype': "stepfilled"}, color=color_a)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlabel('')
    ax3.spines['top'].set_visible(False)
    plt.savefig(os.path.join(f'{savepath}/distribution_{opt.id}_Multi_2D3D_type2.jpg'))

def main(opt):

    os.makedirs(opt.output_path, exist_ok=True)
    src_smi = read_seqs_from_file(opt.src_case_path, datatype='src', src_type=opt.src_type)
    tgt_smi = read_seqs_from_file(opt.tgt_case_path, datatype='target', src_type=opt.src_type)
    prior_samples = read_seqs_from_file(opt.prior_beam_output, datatype='sample', src_type=opt.src_type)
    agent_samples = read_seqs_from_file(opt.agent_beam_output, datatype='sample', src_type=opt.src_type)

    mol = Chem.MolFromSmiles(src_smi[0])
    Draw.MolToFile(mol, f'{opt.output_path}/src.svg')
    cairosvg.svg2png(url=f'{opt.output_path}/src.svg', \
                     write_to=f'{opt.output_path}/src.png')


    mol = Chem.MolFromSmiles(tgt_smi[0])
    Draw.MolToFile(mol, f'{opt.output_path}/tgt.svg')
    cairosvg.svg2png(url=f'{opt.output_path}/tgt.svg', \
                     write_to=f'{opt.output_path}/tgt.png')

    prior_qedlst = []
    prior_2dsimlst = []
    prior_isValidlst = []
    src_lst = []
    tgt_lst = []
    prior_ValidscoreLst = []

    if opt.score_function_type == 'activity_model':
        with open(opt.clf_path, "rb") as f:
            clf = joblib.load(f)
    else:
        clf = None

    for src in src_smi:
        for smi in prior_samples:
            score = calScore(smi, tgt_smi[0], opt.score_function_type, clf)
            # score2 = calc_2D_similarity(smi, tgt_smi[0])
            
            if opt.score_function_type == 'QED_SA':
                score2 = calQED(smi)
            else:
                score2 = calc_2D_similarity(smi, tgt_smi[0])
            prior_qedlst.append(score)
            prior_2dsimlst.append(score2)
            isValid = juiceIsValid(src, smi)
            prior_isValidlst.append(isValid)
            if isValid:
                prior_ValidscoreLst.append(score)
            src_lst.append(src)
            tgt_lst.append(tgt_smi[0])

    agent_qedlst = []
    agent_2dsimlst = []
    agent_isValidlst = []
    agent_ValidscoreLst = []
    for src in src_smi:
        src_new = remove_dummys(src)
        print(f'remove * :{src_new}')
        for smi in agent_samples:
            score = calScore(smi, tgt_smi[0], opt.score_function_type, clf)
            # score2 = calc_2D_similarity(smi, tgt_smi[0])

            if opt.score_function_type == 'QED_SA':
                score2 = calQED(smi)
            else:
                score2 = calc_2D_similarity(smi, tgt_smi[0])
            agent_qedlst.append(score)
            agent_2dsimlst.append(score2)
            isValid = juiceIsValid(src, smi)
            agent_isValidlst.append(isValid)
            if isValid:
                agent_ValidscoreLst.append(score)

    priorZiplst = list(zip(prior_qedlst, prior_2dsimlst, src_lst, prior_samples, prior_isValidlst))
    agentZiplst = list(zip(agent_qedlst, agent_2dsimlst, src_lst, agent_samples, agent_isValidlst))

    draw(prior_ValidscoreLst, agent_ValidscoreLst, opt.output_path, opt)
    draw2(prior_ValidscoreLst, agent_ValidscoreLst, opt.output_path, opt)
    tmp = pd.DataFrame(src_lst)
    tmp.columns = ['src']
    tmp['tgt'] = tgt_lst
    tmp['prior_sample'] = prior_samples
    tmp['prior_qed'] = prior_qedlst
    tmp['prior_2dsim'] = prior_2dsimlst
    tmp['prior_isValid'] = prior_isValidlst

    tmp['agent_sample'] = agent_samples
    tmp['agent_qed'] = agent_qedlst
    tmp['agent_2dsim'] = agent_2dsimlst
    tmp['agent_isValid'] = agent_isValidlst
    tmp.to_csv(os.path.join(f'{opt.output_path}/detail_log.csv'))
    if opt.score_function_type == 'SIM_3D' or opt.score_function_type == 'SIM_3D1' \
        or opt.score_function_type == 'SIM_3D2' or opt.score_function_type == 'QED_SA':
        draw(prior_ValidscoreLst, agent_ValidscoreLst, opt.output_path, opt)
        draw2(prior_ValidscoreLst, agent_ValidscoreLst, opt.output_path, opt)
        drawMulti2D3D(tmp,opt.output_path,opt,0)
        drawMulti2D3D2(tmp,opt.output_path,opt,0)
    # drawTopimol(priorZiplst, opt.topi, opt.output_path, 'prior')
    # drawTopimol(agentZiplst, opt.topi, opt.output_path, 'agent')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='oneCaseData.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-topi', type=int,
                        required=True, default=10,
                        help='save how many smiles to draw mol_figure')
    parser.add_argument('-beam_size', type=int, default=250,
                        help='test smiles size,total numbers')
    parser.add_argument('-id', type=int, default=10, required=True,
                        help='the id of the pair of fragments')
    parser.add_argument('-src_type', type=str,
                        required=True, default='L',
                        help='one case src path.must contain caseId')
    parser.add_argument('-src_case_path', type=str,
                        required=True,
                        help='one case src path.must contain caseId')
    parser.add_argument('-tgt_case_path', type=str,
                        required=True,
                        help='one case tgt path.must contain caseId')
    parser.add_argument('-prior_beam_output', type=str,
                        required=True,
                        help='prior_beam_output txt path.')
    parser.add_argument('-agent_beam_output', type=str,
                        required=True,
                        help='agent_beam_output txt path.')
    parser.add_argument('-output_path', type=str,
                        required=True,
                        help='one case output tgt path.')
    parser.add_argument('-clf_path', type=str,
                        default='./data/clf_jak3_active.pkl',
                        help='one case output tgt path.')
    parser.add_argument('-score_function_type', type=str,
                        required=True,
                        #['QED', 'MW', 'CLOGP', 'NOS', 'SIM_3D', 'tanimoto','activity_model']
                        default='tanimoto',
                        help='one case output tgt path.')
    opt = parser.parse_args()
    main(opt)


