#!/usr/bin/env python
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# This code is an analysis module for WAD-QC 2.0: a server for automated 
# analysis of medical images for quality control.
#
# The WAD-QC Software can be found on 
# https://bitbucket.org/MedPhysNL/wadqc/wiki/Home
# 
#
# Changelog:
#   20200508: dropping support for python2; dropping support for WAD-QC 1; toimage no longer exists in scipy.misc
#   20200225: prevent accessing pixels outside image
#   20190426: Fix for matplotlib>3
#   20181017: fix double occurence of HU High Head
#   20170622: added MeanHigh result, as this gives most deviations
#   20170502: added radiusmm param for air roi location
#   20170227: removed double headHU_pvc definition; error in override_settings (anatomy ignored)
#   20161220: removed testing stuff; removed class variables
#   20161216: added use_anatomy param
#   20160802: sync with pywad1.0
#   20160622: removed adding limits (now part of analyzer)
#   20160620: removed idname; force separate config for head and for body; remove quantity and units
# ./QCCT_wadwrapper.py -c Config/ct_philips_umcu_series_mx8000idt.json -d TestSet/StudyMx8000IDT -r results_mx8000idt.json

__version__ = '20200508'
__author__ = 'aschilham'

import os
# this will fail unless wad_qc is already installed
from wad_qc.module import pyWADinput
from wad_qc.modulelibs import wadwrapper_lib

if not 'MPLCONFIGDIR' in os.environ:
    import pkg_resources
    try:
        #only for matplotlib < 3 should we use the tmp work around, but it should be applied before importing matplotlib
        matplotlib_version = [int(v) for v in pkg_resources.get_distribution("matplotlib").version.split('.')]
        if matplotlib_version[0]<3:
            os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 
    except:
        os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 

import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.

try:
    import pydicom as dicom
except ImportError:
    import dicom
import QCCT_lib
import QCCT_constants as lit

def logTag():
    return "[QCCT_wadwrapper] "

# MODULE EXPECTS PYQTGRAPH DATA: X AND Y ARE TRANSPOSED!

# helper functions
def _getScannerDefinition(params):
    # Use the params in the config file to construct an Scanner object
    try:
        # a name for identification
        scannername = params['scannername']

        # three materials in Philips Performance Phantom Head
        headHU_air   = float(params['headHU_air'])
        headHU_water = float(params['headHU_water'])
        try:
            headHU_pvc   = float(params['headHU_pvc'])
        except:
            headHU_pvc   = float(params['headHU_shell']) # for Siemens scanner, which does not use PVC for shell

        # inner and outer diameter (in mm) of PVC skull (container) of head phantom
        headdiammm_in    = float(params['headdiammm_in'])
        headdiammm_out   = float(params['headdiammm_out'])

        # three materials in Philips Performance Phantom Body
        bodyHU_aculon   = float(params['bodyHU_aculon'])
        bodyHU_teflon   = float(params['bodyHU_teflon'])
        try:
            bodyHU_water    = float(params['bodyHU_water'])
        except:
            bodyHU_water    = float(params['bodyHU_air'])
            
    except AttributeError as e:
        raise ValueError(logTag()+" missing scanner definition parameter!"+str(e))

    return QCCT_lib.Scanner(scannername, 
                            [headHU_air, headHU_water, headHU_pvc],
                            headdiammm_in, headdiammm_out,
                            [bodyHU_aculon, bodyHU_teflon, bodyHU_water])
    
def override_settings(cs, params):
    """
    Look for 'use_' params in to force behaviour of module
    """
    try:
        use_anatomy = params['use_anatomy']
        if 'head' in use_anatomy.lower():
            cs.anatomy = lit.stHead
        elif 'body' in use_anatomy.lower():
            cs.anatomy = lit.stBody
        else:
            raise ValueError('Unknown value %s for param use_anatomy'%use_anatomy)
    except:
        pass

    try:
        cs.forceScanner.HeadAirDistmm = float(params['use_headairdistmm'])
    except:
        pass

    try:
        cs.forceScanner.BodyAirDistmm = float(params['use_bodyairdistmm'])
    except:
        pass

##### Real functions
def qc_series(data, results, action):
    """
    QCCT_UMCU Checks: extension of Philips QuickIQ (also for older scanners without that option), for both Head and Body if provided
      Uniformity
      HU values
      Noise
      Linearity 

    Workflow:
        1. Read image or sequence
        2. Run test
        3. Build xml output
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    dcmInfile, pixeldataIn, dicomMode = wadwrapper_lib.prepareInput(data.series_filelist[0], headers_only=False, logTag=logTag())
    qclib = QCCT_lib.CT_QC()
    cs = QCCT_lib.CTStruct(dcmInfile=dcmInfile, pixeldataIn=pixeldataIn, dicomMode=dicomMode)
    cs.forceScanner = _getScannerDefinition(params)
    cs.verbose = False
    override_settings(cs, params)

    ## id scanner
    error = qclib.DetermineCTID(cs)
    if(error == True or cs.guessScanner.name == lit.stUnknown):
        raise ValueError("{} ERROR! Cannot determine CT ID".format(logTag))

    error = qclib.HeadOrBody(cs)
    if(error == True or cs.anatomy == lit.stUnknown):
        raise ValueError("{} ERROR! Cannot determine Anatomy".format(logTag))

    # only uncomment if same config used for all scanners: idname += cs.guessScanner.name

    ## 2. Run tests
    error = qclib.AnalyseCT(cs)
    if error:
        raise ValueError("{} ERROR! Error in AnalyseCT".format(logTag))

    ## Struct now contains all the results and we can write these to the
    ## WAD IQ database
    includedlist = [
        'skull_avg',
        'roiavg',
        'roisd',
        'snr_hol',
        'unif',
        'linearity',
        'maxdev',
        'shiftxypx',
    ]
    excludedlist = [
        'verbose',
        'dcmInfile',
        'pixeldataIn',
        'dicomMode',
        'hasmadeplots',
        'guessScanner',
        'anatomy',
        'roiavg',
        'roisd',
        'snr_hol',
        'unif',
        'linearity',
        'maxdev',
        'shiftxypx',
        'valid'
        'unif_slice',
        'unif_rois'
    ]
    skull_val = -2024 # low value, will need a higher
    
    # make sure skull_avg is used if valid, but preserve order of results
    for elem in cs.__dict__:
        if elem in includedlist:
            try:
                elemval =  cs.__dict__[elem]
                if 'skull_avg' in elem: # skull_avg only defined for head
                    skull_val  = elemval
                    break
            except:
                print(logTag()+"error for", elem)

    for elem in cs.__dict__:
        if elem in includedlist:
            newkeys = []
            newvals = []
            try:
                elemval =  cs.__dict__[elem]
                if 'roiavg' in elem: # array of avgs
                    newkeys.append('MeanCenter')
                    newvals.append(elemval[0])
                    newkeys.append('MeanAir')
                    newvals.append(elemval[3])
                    if skull_val <= -1024: # skull_avg only defined for head; if not head, then take teflon plug
                        newkeys.append('MeanHigh') 
                        newvals.append(max(elemval))
                        
                elif 'shiftxypx' in elem:
                    newkeys.append('shiftxpx')
                    newvals.append(elemval[0])
                    newkeys.append('shiftypx')
                    newvals.append(elemval[1])
                elif 'skull_avg' in elem:
                    skull_val  = elemval
                    if elemval > -1024: # skull_avg only defined for head; if not head, then take teflon plug
                        newkeys.append('MeanHigh')
                        newvals.append(elemval)
                else:
                    newkeys.append(str(elem))
                    newvals.append(elemval)
            except:
                print(logTag()+"error for",elem)
                elemval = -1.
            for key,val in zip(newkeys,newvals):
                varname = key
                results.addFloat(varname, val)

    ## Build thumbnail
    filename = 'test'+'.jpg' # Use jpg if a thumbnail is desired
    qclib.saveAnnotatedImage(cs, filename)
    varname = 'CTslice'
    results.addObject(varname, filename) 

def acqdatetime_series(data, results, action):
    """
    Read acqdatetime from dicomheaders and write to IQC database

    Workflow:
        1. Read only headers
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    ## 1. read only headers
    dcmInfile = dicom.read_file(data.series_filelist[0][0], stop_before_pixels=True)

    dt = wadwrapper_lib.acqdatetime_series(dcmInfile)

    results.addDateTime('AcquisitionDateTime', dt) 

def header_series(data, results, action):
    """
    Read selected dicomfields and write to IQC database

    Workflow:
        1. Run tests
        2. Build xml output
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    info = 'dicom'
    dcmInfile, pixeldataIn, dicomMode = wadwrapper_lib.prepareInput(data.series_filelist[0], headers_only=True, logTag=logTag())
    qcctlib = QCCT_lib.CT_QC()
    cs = QCCT_lib.CTStruct(dcmInfile=dcmInfile, pixeldataIn=pixeldataIn, dicomMode=dicomMode)
    cs.verbose = False
    override_settings(cs, params)

    error = qcctlib.HeadOrBody(cs)
    if(error == True or cs.anatomy == lit.stUnknown):
        raise ValueError("{} ERROR! Cannot determine Anatomy".format(logTag))

    result_dict = {}
    # only uncomment if same config used for all scanners: idname += "_"+cs.guessScanner.name

    ## 1. Run tests
    dicominfo = qcctlib.DICOMInfo(cs,info)

    ## 2. Add results to 'result' object
    # plugionversion is newly added in for this plugin since pywad2
    varname = 'pluginversion'
    results.addString(varname, str(qcctlib.qcversion))
    varname = 'Anatomy'
    results.addString(varname, str(cs.anatomy))
    for di in dicominfo:
        varname = di[0]
        results.addString(varname, str(di[1])[:min(len(str(di[1])),100)])

if __name__ == "__main__":
    data, results, config = pyWADinput()

    # read runtime parameters for module
    for name,action in config['actions'].items():
        if name == 'acqdatetime':
            acqdatetime_series(data, results, action)

        elif name == 'header_series':
            header_series(data, results, action)
        
        elif name == 'qc_series':
            qc_series(data, results, action)

    #results.limits["minlowhighmax"]["mydynamicresult"] = [1,2,3,4]

    results.write()
    