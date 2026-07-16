# reference

import os
import re 

arxiv_path = 'paper/arxiv'

# remove the arxiv directory if it exists
if os.path.exists(arxiv_path):
    os.system(f'rm -rf {arxiv_path}')

if os.path.exists('paper/arxiv-submission'):
    os.system(f'rm -rf paper/arxiv-submission')


# create directory paper/arxiv if it doesn't exist
if not os.path.exists(arxiv_path):
    os.makedirs(arxiv_path)

copyfiles = ['main.tex', 'community-detection-learning-progress.tex', 'synthetic-community-heatmaps.tex', 'macros.tex', 'refs.bib']

for f in copyfiles:
    if not os.path.exists(f'paper/{f}'):
        raise FileNotFoundError(f'File paper/{f} does not exist. Please make sure it is present before running this script.')
    os.system(f'cp paper/{f} {arxiv_path}/{f}')

# read in the tex file
tex = open(f'{arxiv_path}/main.tex', 'r').read()

# find all instances of \includegraphics in tex

# find and remove all instances of "../fig/" in tex 

tex = tex.replace('../fig/', '')

graphics = re.findall(r'\\includegraphics\[.*?\]\{(.*?)\}', tex)

for graphic in graphics: 
    if not os.path.exists(f'fig/{graphic}'):
        raise FileNotFoundError(f'Graphic {graphic} does not exist. Please make sure it is present before running this script.')
    os.system(f'cp fig/{graphic} {arxiv_path}/{graphic}')


tex += "\n\n\\typeout{get arXiv to do 4 passes: Label(s) may have changed. Rerun}"

# write tex to main.tex
with open(f'{arxiv_path}/main.tex', 'w') as f:
    f.write(tex)
    
# run latexmk on arxiv_path/main.tex 

# os.system(f'cd {arxiv_path} && latexmk -pdf main.tex')

# clean up (remove .pdf if you want to check the files before zipping)
for suffix in ['.aux', '.fdb_latexmk', '.fls', '.log', '.out', '.synctex.gz', '.tdo', '.blg', '.pdf']:
    # delete all files in arxiv_path with the suffix
    for f in os.listdir(arxiv_path):
        if f.endswith(suffix):
            os.remove(f'{arxiv_path}/{f}')
            
# zip the arxiv path directory

os.system(f'cd {arxiv_path} && zip -r ../arxiv-submission.zip *')

print("testing that the arxiv-submission.zip file can be unzipped and compiled correctly.")
os.system(f'cd paper && unzip arxiv-submission.zip -d arxiv-submission-test')
os.system(f'cd paper/arxiv-submission-test && latexmk -pdf main.tex')
print("Please check the file arxiv-submission-test/main.pdf")
print("If this file looks good, the zip file arxiv-submission.zip is likely ready for submission.")




# delete the arxiv_path directory (optional, remove for troubleshooting)
# os.system(f'rm -rf {arxiv_path}')
    
