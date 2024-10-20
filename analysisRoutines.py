from __future__ import print_function
from scipy.interpolate import interp1d
import scipy
from orphics import maps,io,cosmology,catalogs,stats # msyriac/orphics ; pip install -e . --user
from pixell import enmap,reproject,curvedsky,enplot,utils
import numpy as np
import tqdm

class settingsClass(object):
    """ 
    An object with all the settings for the analysis.
    
    mask           - The apodized mask used for SHTs
    mask_inpt_sub  - The mask of inpainted sources and regions to avoid. Binary.

    "verbose":False,                - To print more or less outputs
    'output_shape':mask.shape,      - The shape of the maps.
    'output_wcs':mask.wcs,          - The astropy wcs of the maps.
    'LMIN_highpass':2500,           - Modes above which to keep in the high pass map
    'LMAX_highpass':16000,          - Modes to cut off from the high pass filtered map. 16000 is the maximum scale in the ILC maps
    'transition_filter_width':150,  - The width of the cosine filter taper. So the filter smoothly transitions from 0 to 1
    'LMIN_lowpass':0,               - Modes above which to keep in the low pass map
    'LMAX_lowpass':2000,            - Modes below which to keep in the low pass map
    "useCentralSign":False          - The estimator uses the sign of the low pass map. This can either be applied for each pixel, or the sign for the profile can be set by the central value. Default is the map level. Using the profile level was found to reduce correlations between objects for very dense samples.
    'isNullTest':False,             - Used to randomize the signs. This will kill the signal and so enables a simple null test.
    'pixelWindowCorr':False,        - Apply a pixel window correction
    'noLowpassFilter':False,        - Option to use all the modes in the "low pass" map. Used for testing.

    *initial_data  -
    **kwargs       - a dictionary of key, values to set
    """
    def __init__(self,mask,mask_inpt_sub, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
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
        'pixelWindowCorr':False,
        'noLowpassFilter':False,
        }
        self.mask = mask
        self.mask_inpt_sub = mask_inpt_sub
        for key in defaults:
            if not hasattr(self, key):
                setattr(self,key, defaults[key])
        if not hasattr(self, 'mask'):
            raise AssertionError("Need to specify 'mask'. ")

        shape,wcs = mask.geometry

        if not hasattr(self, 'mask_inpt_sub'):
            raise AssertionError("Need to specify 'mask_inpt_sub; of points to avoid: ")


def saveRes(fileName,data_dic,stacks):
    """
    Save the 1D and 2D stack to a file.

    filename - path to the file
    data_dic - a orphics results dictionary
    stacks - the 2D stacks image.
    """
    res_dic = {}
    weights_dic = {}
    stacks_dic = {}
    sign_dic = {}
    for key in data_dic.keys():
        res_dic[key]=data_dic[key].vectors['i1d']
        weights_dic[key]=data_dic[key].vectors['weight']
        sign_dic[key]=data_dic[key].vectors['sign']
        stacks_dic[key] = np.array(stacks[key])
        if key+'breakdown'  in stacks.keys():
            stacks_dic[key+'breakdown']=stacks[key+'breakdown']
    np.savez(fileName,i1d=res_dic,weights=weights_dic,stacks_dic=stacks_dic,sign_dic=sign_dic)
    print(f'Saved to {fileName}')

def filter_map(input_map,mask,filt,inverseCfilter=False,cFilter=False,cl_thr=None,LMAX_FILT=None,verbose=False,pixelWindowCorr=False):
    """
    Filter a map in harmonic space. Used for high pass and low pass filtering the data

    input map - the map to be filtered
    mask - a mask. This is applied to the map before the SHT operations
    filt - an array of the filter to be applied to each l mode
    
    Optional

    inverseCfilter - use an additional 1/C_ell filter
    cFilter - use an additional C_ell filter
    cl_thr - if specified a theory C_ell will be used for the above operations. Otherwise it is the emperical C_ell
    LMAX_FILT - the maximum scale used for the filter. Modes above this are set to 0
    pixelWindowCorr - If you want to remove the pixel window function
    verobse - Print which steps are being run
    """
    if verbose: print('shting')

    input_map = input_map*mask
    if pixelWindowCorr:
        print("Removing map's pixel window function")
        input_map = enmap.apply_window(input_map, pow=-1.0)
    if LMAX_FILT is not None:
        alms = curvedsky.map2alm(input_map,lmax=LMAX_FILT)
    else:
        alms = curvedsky.map2alm(input_map,lmax=20000)
    if verbose: print('shtd')
    filt = filt.copy()
    filt[~np.isfinite(filt)] = 0
    print(alms.shape,input_map.shape,mask.shape)
    if inverseCfilter:
        print('using invC weighting')
        if cl_thr is None:
            cl_tmp = curvedsky.alm2cl(alms).reshape(-1)
        else:
            cl_tmp = cl_thr.copy()
        cl_tmp[cl_tmp==0] = np.inf
        if filt.shape[-1]<=cl_tmp.shape[-1]:
            cl_tmp=cl_tmp[:filt.shape[-1]]
            filt/=cl_tmp
        else:
            
            filt[:cl_tmp.shape[-1]]/=cl_tmp
    if cFilter:
        print('using C weighting')
        if cl_thr is None:
            cl_tmp = curvedsky.alm2cl(alms).reshape(-1)
        else:
            cl_tmp = cl_thr.copy()
        if filt.shape[-1]<=cl_tmp.shape[-1]:
            cl_tmp=cl_tmp[:filt.shape[-1]]
            filt*=cl_tmp
        else:
            filt[:cl_tmp.shape[-1]]*=cl_tmp

    filt[:2]=0 
    alms = curvedsky.almxfl(alms,filt)#/norm
        
    if verbose: print('shting')
    return curvedsky.alm2map(alms,mask.copy()*0)


def loadAndFilter(settings,fileName,fileName_lowpass=None):
    """ 
    Load the maps and apply the high and low pass filters

    settings - a settings object with the filter settings
    fileName - the map to be filter

    fileName_lowpass - if supplied the low pass filtering will be applied to this map. Otherwise the low and high pass filtered maps are the same

    Returns: low and high pass filtered maps.

    """
    if settings.verbose: print('reading map')
    mock_map=0
    input_map = enmap.read_map(fileName)
    if settings.verbose: print('maps Read',input_map.shape)

    ells = np.arange(30000)
    filt = np.ones(30000)
    lbeam = np.ones(30000)

    ibeam = lbeam.copy()
    ibeam[ibeam==0]= np.inf
    ibeam = 1/ibeam

    filt[ells<settings.LMIN_highpass-settings.transition_filter_width] = 0
    ell_taper_width = settings.transition_filter_width
    filt[settings.LMIN_highpass-ell_taper_width:settings.LMIN_highpass]*=np.sin(np.arange(ell_taper_width)*np.pi/(2*ell_taper_width))
    filt*=ibeam
    if settings.LMAX_highpass is not None:
        filt[ells>settings.LMAX_highpass]=0


    input_map = enmap.extract(input_map,settings.output_shape,settings.output_wcs)
    if len(input_map.shape)>2:
        print(input_map.shape)
        Nx,Ny = input_map.shape[-2:]
        input_map = input_map.reshape([Nx,Ny])

    if settings.verbose: print('filtering')
    highpass_map = filter_map(input_map.copy(),settings.mask,filt,LMAX_FILT=settings.LMAX_highpass,verbose=settings.verbose,pixelWindowCorr=settings.pixelWindowCorr)

    LMAX_lowpass = settings.LMAX_lowpass
    LMIN_lowpass = settings.LMIN_lowpass
    if LMAX_lowpass is None:
        LMAX_lowpass = settings.LMIN_highpass-min(250, 10+2*settings.transition_filter_width)
    if LMIN_lowpass is None:
        LMIN_lowpass = 20

    assert(LMAX_lowpass<settings.LMIN_highpass)
    filt = np.ones(30000)
    filt*=ibeam
    filt[ells<LMIN_lowpass] = 0

    filt[ells>LMAX_lowpass+settings.transition_filter_width] = 0
    ell_taper_width = settings.transition_filter_width
    filt[LMAX_lowpass:LMAX_lowpass+ell_taper_width]*=np.cos(np.arange(ell_taper_width)*np.pi/(2*ell_taper_width))

    if fileName_lowpass is not None:
        print('loading seperate large scales')
        print(fileName_lowpass)
        input_map = enmap.read_map(fileName_lowpass)
    if not settings.noLowpassFilter:
        lowpass_map = filter_map(input_map,settings.mask,filt,LMAX_FILT=LMAX_lowpass+500,verbose=settings.verbose)
    else:
        lowpass_map = input_map 

    return lowpass_map,highpass_map




def runStack(settings,lowpass_map,tau_map,ras,decs,width=10*utils.arcmin,random=False,n_random_repeats=1,weight_threshold=0):
    """
    Compute the stacked profiles for a set of objects.

    Settings - the settings containing the binary mask
    lowpass_map - the large scale lowpass cmb map
    tau_map - the high pass filtered cmb map
    ras,decs - list of coordinates to stack on.

    optional
    width of the patched to stack on.
    random - stack on random locations away from masked regions.
    n_random_repeats - Run the random catalogs at the n_random_repeats.
    weight_threshold - Skip locations where the  value of the large scale field is less than this. 
    
    Returns:
    cents, profiles, stacks - containing the location of the radial samples, the 1D profiles and the 2D stacks.
    
    """
    #
    if not settings.useCentralSign: 
        tau_map*=np.sign(lowpass_map)
    bmask = settings.mask.copy()
    bmask[bmask<0.99] = 0
    # Initialize stacks
    i = 0
    istack = 0
    # 1d statistics collector
    s = stats.Stats()
    ras_tmp = np.array([])
    decs_tmp = np.array([])

    cntr_all=0
    weight_tot = 0
    icut = None
    dec,ra = tau_map.pix2sky(np.array(tau_map.shape[-2:])//2)
    tmp = reproject.thumbnails(tau_map, [(dec,ra)],res=None,r=width+2*utils.arcmin,oversample=1,order=1,pixwin=True)
    pix = tmp.shape[-1]
    for II,(ra,dec) in tqdm.tqdm(enumerate(zip(ras,decs))):

        mcut = reproject.cutout(bmask, ra=np.deg2rad(ra), dec=np.deg2rad(dec),npix=pix)
        if mcut is None: 
            continue
        if np.any(mcut)<=0: 
            continue
            

        px,py = bmask.sky2pix([np.deg2rad(dec),np.deg2rad(ra)]).astype('int')
        

        # For random stacking
        if random==1:
            cntr=0
            while 1:
                if cntr==100: break
                px = np.random.randint(0,bmask.shape[0])
                py = np.random.randint(0,bmask.shape[1])
                dec_test,ra_test = bmask.pix2sky(np.array([px,py]))
                mcut = reproject.cutout(bmask, ra=np.deg2rad(ra_test), dec=np.deg2rad(dec_test),npix=pix)
                if mcut is None: 
                    continue

                cntr+=1
                if np.any(mcut)<=0: 
                    continue
                    
                px,py = bmask.sky2pix([np.deg2rad(dec_test),np.deg2rad(ra_test)]).astype('int')

                if settings.mask_inpt_sub[px,py]==0: continue
                break 
            if cntr==100: continue 
            ra= ra_test
            dec = dec_test
        # It not random and point is masked skip it.
        elif settings.mask_inpt_sub[px,py]==0: 
            continue
        # Stamp cutouts
        
        #lowpass_map
        icut=  reproject.thumbnails(tau_map, [(np.deg2rad(dec),np.deg2rad(ra))],res=None,r=width+2*utils.arcmin,oversample=1,order=1,pixwin=False)[0]#.shape
        weight = (lowpass_map[px,py])
        if  np.abs(weight)<weight_threshold: 
            continue
        
        # Some binning tools
        if i==0:
            modrmap = np.rad2deg(icut.modrmap())*60. # Map of distance from center
            bin_edges = np.linspace(0,10,10)
            binner = stats.bin2D(modrmap,bin_edges)

        # tau map is input. Only reweight for random maps or null test. This effectively randomizes the large scales and kills the signal.
        if random==2 or settings.isNullTest:
            if np.random.uniform(-1,1)<0:
                weight=-weight
        try:
            #cents,i1d = binner.bin(icut)
            if settings.useCentralSign or random==2 or settings.isNullTest: 
                cents,i1d = binner.bin(icut*np.sign(weight))
            else:
                cents,i1d = binner.bin(icut)
        except IndexError:
            print(f'Issue with ra,dec {ra},{dec} ')
            print(np.shape(icut),np.shape(istack))
            continue
        s.add_to_stats("i1d",i1d)
        s.add_to_stats('weight',np.array(np.atleast_1d(np.mean(np.abs(weight)))))
        s.add_to_stats('sign',np.array(np.atleast_1d((np.sign(weight)))))
        # Stack
        istack = istack + icut*np.sign(weight)
        i +=1 
        cntr_all+=1
        weight_tot+=np.abs(weight)
        if cntr_all%200==0 and settings.verbose: print(cntr_all)
    print(cntr_all,weight_tot,i)
    s.get_stats()
    istack = istack / i 
    return cents,s,istack



def runEst(fileName_highpass,ras,decs,settings,suffix='',res = {},stacks = {},doRandom=1,fileName_lowpass=None):
    """
    Run the estimator

    fileName_highpass - The file to load and high pass filter. This is also used for the low pass if fileName_lowpass is not passed
    ras               - The ras of the objects
    decs              - The decs of the objects
    settings          - The settings object.

    Options - 

    res = {}  - stored the 1D profiles in this object
    stacks = {} - store the 2D stacks in this object
    suffix - ' a tag for the data used in dictionary of the results.
    doRandom - do Random or just the signal
    fileName_lowpass - The file for the low pass map.
    
    """

    print(f' lmin_highpass:{settings.LMIN_highpass},\n lmax_highpass:{settings.LMAX_highpass}') 
    print(f' LMIN_lowpass:{settings.LMIN_lowpass},\n LMAX_lowpass:{settings.LMAX_lowpass}, \n transition_filter_width:{settings.transition_filter_width}') 
    print(f'  pixelWindowCorr:{settings.pixelWindowCorr}')
    print(f' isNullTest:{settings.isNullTest}')
    print(fileName_highpass)
    
    
    key = f'tau{suffix}'
    
    
    lowpass_map,tau_map = loadAndFilter(settings,fileName=fileName_highpass,fileName_lowpass=fileName_lowpass)

    if doRandom:
        cents,stat_rand,stack_rand =  runStack(settings,lowpass_map,tau_map,ras,decs,random=doRandom)
        res[key+'-rand']=stat_rand
        stacks[key+'-rand']=stack_rand
    else:

        cents,stat,stack =  runStack(settings,lowpass_map,tau_map,ras,decs,random=False)
        res[key]=stat
        stacks[key]=stack
    
    print(f"base: {key}")
    return cents,res,stacks
