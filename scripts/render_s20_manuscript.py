"""Match the inspected original DOCX S20 (image22.tif): four stacked wide panels."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from asc_reproduction.provenance import file_hash, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.source/'COMPLETE.json').read_text())
    if manifest['status']!='succeeded' or manifest['completed_folds']!=500:
        raise ValueError('Incomplete source')
    models=['SNV-TOP','CNV-gene','CNV-arm','CNV-cluster']
    tables=[]; hashes={}
    for model in models:
        name=f'{model}_top20_seed_means.csv'
        if file_hash(args.source/name)!=manifest['artifacts'][name]: raise ValueError('Changed source')
        table=pd.read_csv(args.source/name,index_col=0)
        if table.shape!=(20,100) or table.columns.tolist()!=[str(i) for i in range(100)]:
            raise ValueError('Unexpected panel shape/seed order')
        tables.append(table);hashes[name]=file_hash(args.source/name)
    args.output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7})
    # Original embedded TIFF is 7500 x 10432: preserve its page-shaped aspect.
    fig=plt.figure(figsize=(10,10*10432/7500))
    positions=[.765,.512,.266,.020]
    for letter,table,y in zip('ABCD',tables,positions):
        ax=fig.add_axes([.10,y,.80,.208])
        cax=fig.add_axes([.915,y,.015,.208])
        sns.heatmap(table,ax=ax,cbar_ax=cax,cmap='viridis',vmin=0,
                    xticklabels=False,yticklabels=True)
        ax.set_xlabel('Seed',labelpad=2);ax.set_ylabel('Feature',labelpad=1)
        ax.tick_params(axis='y',labelrotation=0,labelsize=7,length=2,pad=1)
        cax.tick_params(labelsize=7,length=2)
        fig.text(.015,y+.208,letter,fontfamily='DejaVu Serif',fontsize=25,va='bottom')
    fig.savefig(args.output/'S20_manuscript_layout.png',dpi=180)
    path=args.output/'S20_manuscript_layout.tiff'
    fig.savefig(path,dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    with Image.open(path) as im: rgb=im.convert('RGB')
    rgb.save(path,dpi=(600,600),compression='tiff_lzw')
    write_json(args.output/'COMPLETE.json',{'status':'succeeded','model_fitting':False,
        'reference':'20260922 ASC_WES_supplementary.docx / word/media/image22.tif',
        'reference_dimensions':[7500,10432],'layout':'four vertically stacked wide panels A-D',
        'source_hashes':hashes,'script_sha256':file_hash(Path(__file__)),
        'artifacts':{p.name:file_hash(p) for p in args.output.iterdir() if p.is_file()}})


if __name__=='__main__':main()
