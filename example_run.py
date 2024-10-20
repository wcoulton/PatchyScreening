from analysisRoutines import *

from orphics import maps,io,cosmology,catalogs,stats # msyriac/orphics ; pip install -e . --user
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--totSplits',dest='totSplits',type=int,default=0)
parser.add_argument('--splitNum',dest='splitNum',type=int,default=0)
args = parser.parse_args()

print("set your data path!")
dataDir = '/path/To/Your/Data/'

print('reading mask')

saveStackDir ='./'

mask = enmap.read_map(dataDir+'/outputMask_wide_mask_GAL070_apod_1.50_deg_wExtended.fits')
mask_inpt_sub  = enmap.read_map(dataDir+'/inpainted_and_subtracked_sources.fits')


defaults = {
"verbose":False,
'output_shape':mask.shape,
'output_wcs':mask.wcs,
'LMIN_highpass':2500,
'LMAX_highpass':16000,
'transition_filter_width':150,
'LMIN_lowpass':0,
'LMAX_lowpass':2000,
'isNullTest':False,
'pixelWindowCorr':True,
"useCentralSign": False,
}

settings = settingsClass(mask=mask,mask_inpt_sub=mask_inpt_sub,**defaults)

print('loading galaxy cats')
def get_catalog():
    cols = catalogs.load_fits(dataDir+'blue_fullsky_catalogs_v2.fits',['ra','dec'])

    ras = cols['ra'][cols['dec']<22.]
    decs = cols['dec'][cols['dec']<22.]
    return ras,decs

ras,decs = get_catalog()
suffix=''
# Either the stacking routine can be parralized or the catalog could be split as hereg.
if args.totSplits>1:
    nEls = ras.shape[0]
    nSplits = args.totSplits
    splitNum = args.splitNum
    ras = ras[splitNum::nSplits]
    decs = decs[splitNum::nSplits]
    print(f'nSplits: {nSplits} \n splitNum:{splitNum}')
    print(r' N objects: ',ras.shape)
    suffix+=f'_splt_{args.splitNum}_of_{args.totSplits}'
res_wSZ_north = {}
stacks_wSZ_north = {}
print('Running estimator')  


filename_nilc_cmb = f"{dataDir}/ilc_fullRes_TT.fits"#_noKspaceCor
fileName_lowpass = filename_nilc_cmb

#bin_edges = np.linspace(0,10,10)
#cents =(bin_edges[1:]+bin_edges[:-1])/2.
doRandom = False
if not doRandom:
    cents, _,_= runEst(filename_nilc_cmb,ras,decs,settings,suffix='-unWISE_blue',doRandom=False,res=res_wSZ_north,stacks=stacks_wSZ_north,fileName_lowpass=fileName_lowpass)



    saveRes(saveStackDir+f'test{suffix}',res_wSZ_north,stacks_wSZ_north)

if doRandom:


    res_wSZ_north = {}
    stacks_wSZ_north = {}
    print('Running estimator')  

    cents, _,_= runEst(filename_nilc_cmb,ras,decs,settings,suffix='-unWISE_blue',doRandom=doRandom,res=res_wSZ_north,stacks=stacks_wSZ_north,fileName_lowpass=fileName_lowpass)

    saveRes(saveStackDir+f'test_wRandom{suffix}',res_wSZ_north,stacks_wSZ_north)

    print('done')

