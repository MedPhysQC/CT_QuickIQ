# -*- coding: utf-8 -*-
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
from __future__ import print_function

"""
Warning: THIS MODULE EXPECTS PYQTGRAPH DATA: X AND Y ARE TRANSPOSED!

Changelog:
    20171116: fix scipy version 1.0
    20170622: more extreme avg_skull value to allow reporting for body
    20170502: added radiusmm param for air roi location; added thumbnail with ROIs
    20161220: removed testing stuff; removed class variables
    20161216: allow manually supplied anatomy
    20160902: sync with wad2.0; Unified pywad1.0 and wad2.0
    20150701: updated for iPatient iCT: body and head; other tags; other body
    20150409: Removed scanner definitions; should be passed to cs or in config
    20141009: Update to use dicomMode instead of mode2D
    20140528: Initialize all items of CTStruct in __init__ (bug for gui)
    20140425: Bugfix FindCenterShift
    20140414: Removed 's in DICOM tag name to avoid sql problems
    20140409: Initial split of gui/lib for pywad

"""
__version__ = '20171116'
__author__ = 'aschilham'

import copy
try:
    # wad2.0 runs each module stand alone
    import QCCT_constants as lit
except ImportError:
    from . import QCCT_constants as lit

# First try if we are running wad1.0, since in wad2 libs are installed systemwide
try: 
    # try local folder
    import wadwrapper_lib
except ImportError:
    # try pyWADlib from plugin.py.zip
    try: 
        from pyWADLib import wadwrapper_lib

    except ImportError: 
        # wad1.0 solutions failed, try wad2.0 from system package wad_qc
        from wad_qc.modulelibs import wadwrapper_lib

import numpy as np
from scipy import stats
import scipy.ndimage
import matplotlib.pyplot as plt
from PIL import Image # image from pillow is needed
from PIL import ImageDraw # imagedraw from pillow is needed, not pil
import scipy.misc
# sanity check: we need at least scipy 0.10.1 to avoid problems mixing PIL and Pillow
scipy_version = [int(v) for v in scipy.__version__ .split('.')]
if scipy_version[0] == 0:
    if scipy_version[1]<10 or (scipy_version[1] == 10 and scipy_version[1]<1):
        raise RuntimeError("scipy version too old. Upgrade scipy to at least 0.10.1")

class Scanner:
    def __init__ (self, _name,_HeadGT,_innerdiam,_outerdiam, _BodyGT):
        self.name = _name
        self.HeadGT = copy.deepcopy(_HeadGT) # HU for Air, Water, Teflon
        self.innerskulldiammm = _innerdiam
        self.outerskulldiammm = _outerdiam
        self.BodyGT = copy.deepcopy(_BodyGT)
        self.HeadAirDistmm = 79.75 # distance to center of Air roi for head
        self.BodyAirDistmm = 115.75 # distance to center of Air roi for body

class CTStruct:
    def __init__ (self,dcmInfile,pixeldataIn,dicomMode):
        self.verbose = False

        # input image
        self.dcmInfile = dcmInfile
        self.pixeldataIn = pixeldataIn
        self.dicomMode = dicomMode

        # for matlib plotting
        self.hasmadeplots = False

        # identification
        self.guessScanner = None
        self.forceScanner = None # guessScanner is old way, forceScanner passes scanner definition to struct for usage
        self.anatomy = lit.stUnknown
        
        # measurements for head and body
        self.roiavg = [] # Average HU in rois
        self.roisd  = -1. # SD in center
        self.snr_hol = -1. # (avg+1000)/sd
        self.unif    = -1.
        self.linearity = -1.
        self.maxdev = -1.
        self.shiftxypx = []
        self.valid = False

        # measurements for head only
        self.skull_avg = -2024. # Avg HU in skull

        # for gui
        self.unif_slice = 0
        self.unif_rois = [] # x0,y0,rad


class CT_QC:
    def __init__(self):
        self.qcversion = __version__
        self.sigma_ext = 1.5 # gaussian blur factor for extreme finder

    def readDICOMtag(self,cs,key,imslice=0): # slice=2 is image 3
        value = wadwrapper_lib.readDICOMtag(key,cs.dcmInfile,imslice)
        return value

    def readDICOMspacing(self,cs):
        """
        Try to read it from proper DICOM field. If not available,
        calculate from distance between two slices, if 3D
        """
        key = "0018,0088" # "Spacing Between Slices", // Philips
        value = self.readDICOMtag(cs,key)
        if cs.dicomMode == wadwrapper_lib.stMode2D:
            return value
        if(value != ""):
            return value

        key = "0020,1041" #"Slice Location",
        val1 = self.readDICOMtag(cs,key,imslice=0)
        val2 = self.readDICOMtag(cs,key,imslice=1)
        if(val1 != "" and val2 != ""):
            value = val2-val1
        return value


#----------------------------------------------------------------------
    def DetermineCTID(self,cs):
        error = True
        if not cs.forceScanner is None:
            cs.guessScanner = cs.forceScanner
            error = False
            return error

        cs.guessScanner = Scanner(lit.stUnknown, [0,0,0],0,0,[0,0,0])
        return error

    def HeadOrBody(self,cs):
        error = True
        if not cs.anatomy is lit.stUnknown:
            return False

        cs.anatomy = lit.stUnknown
        for tag in ["0018,1030", "0008,103e"]: #Protocol Name, Series Description (for iPatient) and Force
            dicomvalue = self.readDICOMtag(cs,tag)
            dicomvalue = str(dicomvalue).lower()
            if dicomvalue == 'unknown':
                continue

            if(dicomvalue.find("head")>-1):
                cs.anatomy = lit.stHead
                error = False
            if(dicomvalue.find("body")>-1):
                cs.anatomy = lit.stBody
                error = False
            if error == False:
                break

        return error

    def pix2phantommm(self, cs, pix):
        if cs.dicomMode == wadwrapper_lib.stMode2D:
            pix2mmm = cs.dcmInfile.PixelSpacing[0]
        else:
            pix2mmm = cs.dcmInfile.info.PixelSpacing[0]
        return pix*pix2mmm

    def phantommm2pix(self, cs, mm):
        if cs.dicomMode == wadwrapper_lib.stMode2D:
            pix2mmm = cs.dcmInfile.PixelSpacing[0]
        else:
            pix2mmm = cs.dcmInfile.info.PixelSpacing[0]
        return mm/pix2mmm

#----------------------------------------------------------------------
    def AnalyseCT(self, cs):
        # id scanner (for base values)
        error = self.DetermineCTID(cs)
        if(error == True):
            print("[AnalyzeCT] cannot determine CT ID")
            return error

        # id anatomy of phantom (for test and ROI locations)
        error = self.HeadOrBody(cs)
        if(error == True):
            print("[AnalyzeCT] cannot determine Anatomy")
            return error

        if not cs.dicomMode == wadwrapper_lib.stMode2D:
            dep = np.shape(cs.pixeldataIn)[0]
            cs.unif_slice = int((dep-1)/2) # take middle slice

        if(cs.anatomy == lit.stHead):
            error = self.AnalyseCTHead(cs)
        elif(cs.anatomy == lit.stBody):
            error = self.AnalyseCTBody(cs)

        return error

    def AnalyseCTHead(self, cs):
        """
        Calculate uniformity, water#, noise, CT#, shift
        1. Find fantom center shift
        2. Find mean and SD in relevant ROIs
        3. Find mean and SD in skull
        4. Find
        """
        error = True
        # 1. find center shift
        if cs.dicomMode == wadwrapper_lib.stMode2D:
            error,cs.shiftxypx = self.FindCenterShift2D(cs,cs.pixeldataIn,matter=lit.stAir)
        else:
            error,cs.shiftxypx = self.FindCenterShift2D(cs,cs.pixeldataIn[cs.unif_slice],matter=lit.stAir)

        if(error):
            print("[AnalyseCTHead] cannot findcentershift")
            return error

        # 2. find mean and SD in relevant regions
        #   1    3
        #
        #   0  2
        #
        midx     = int(self.phantommm2pix(cs,125.)+.5)
        stepx    = int(self.phantommm2pix(cs,65.)+.5)
        roidim   = int(self.phantommm2pix(cs,18.)+.5) # 25 for body // IEC say at least 10% of diam (178mm)
        sddim    = 4*roidim #IEC say at least 40% of diam (178mm) and may not overlap with other structures
        diagstep = int(self.phantommm2pix(cs,cs.guessScanner.HeadAirDistmm)+.5) # distance to Air ROI
        roirad   = roidim/2.

        roix = []
        roiy = []
        roix.append(cs.shiftxypx[0]+midx-int(roidim/2)) # center
        roix.append(roix[0])            # 12 o'clock
        roix.append(roix[0]+stepx)      #  3 o'clock
        roix.append(roix[0]+diagstep)  #
        roix.append(cs.shiftxypx[0]+midx-int(sddim/2)) # center

        roiy.append(cs.shiftxypx[1]+midx-int(roidim/2)) # center
        roiy.append(roiy[0]-stepx)            # 12 o'clock
        roiy.append(roiy[0])                  #  3 o'clock
        roiy.append(roiy[0]-diagstep)  #
        roiy.append(cs.shiftxypx[1]+midx-int(sddim/2)) # center

        # first avg
        cs.roiavg = []
        for kk in range(0,len(roix)-1):
            sumv = 0.;
            count = 0
            for ky in range(0,roidim):
                for kx in range(0,roidim):
                    if( (kx-roirad)**2+(ky-roirad)**2<roirad**2 ):
                        if cs.dicomMode == wadwrapper_lib.stMode2D:
                            sumv += cs.pixeldataIn[roix[kk]+kx,roiy[kk]+ky]
                        #    self.pixeldataIn[roix[kk]+kx,roiy[kk]+ky] = -1000
                        else:
                            sumv += cs.pixeldataIn[cs.unif_slice,roix[kk]+kx,roiy[kk]+ky]
                        # self.pixeldataIn[iz,roix[kk]+kx,roiy[kk]+ky] = -1000
                        count += 1
            if(count>0):
                cs.roiavg.append(sumv/count)
            else:
                cs.roiavg.append(0)
            # for gui
            cs.unif_rois.append([roix[kk]+roidim/2,roiy[kk]+roidim/2, roidim/2])

        # now SD
        cs.roisd = 0.;
        sdrad = sddim/2.
        sumv = np.float32(0.);
        sumv2 = np.float32(0.)
        count = 0
        np.seterr(over='raise')
        for ky in range(0,sddim):
            for kx in range(0,sddim):
                if( (kx-sdrad)**2+(ky-sdrad)**2<sdrad**2 ):
                    if cs.dicomMode == wadwrapper_lib.stMode2D:
                        val = cs.pixeldataIn[roix[-1]+kx,roiy[-1]+ky]
                    else:
                        val = cs.pixeldataIn[cs.unif_slice,roix[-1]+kx,roiy[-1]+ky]
                    sumv += val
                    try:
                        sumv2 += val*val
                    except:
                        print("[AnalyseCTHead] ERROR!",type(val),type(sumv),type(sumv2),val,sumv,sumv2)
                    #  self.pixeldataIn[iz,roix[kk]+kx,roiy[kk]+ky] = -1000
                    count += 1
        if(count>1):
            cs.roisd = (count*sumv2-sumv*sumv)/count
            cs.roisd = np.sqrt(cs.roisd/(count-1.))

        # for gui
        cs.unif_rois.append([roix[-1]+sddim/2,roiy[-1]+sddim/2, sddim/2])

        # now avg skull
        cs.skull_avg = 0.
        innerroidim = int(self.phantommm2pix(cs,cs.guessScanner.innerskulldiammm)+.5)
        outerroidim = int(self.phantommm2pix(cs,cs.guessScanner.outerskulldiammm)+.5)
        roix = []
        roiy = []
        roix.append(cs.shiftxypx[0]+midx-int(outerroidim/2)) # center
        roix.append(cs.shiftxypx[0]+midx-int(innerroidim/2)) # center
        roiy.append(cs.shiftxypx[1]+midx-int(outerroidim/2)) # center
        roiy.append(cs.shiftxypx[1]+midx-int(innerroidim/2)) # center
        outerroirad = outerroidim/2.
        innerroirad = innerroidim/2.
        sumv = 0.;
        count = 0
        for ky in range(0,outerroidim):
            for kx in range(0,outerroidim):
                rrr2 = (kx-outerroirad)**2+(ky-outerroirad)**2
                if( rrr2>innerroirad**2 and rrr2<outerroirad**2):
                    if cs.dicomMode == wadwrapper_lib.stMode2D:
                        sumv += cs.pixeldataIn[roix[0]+kx,roiy[0]+ky]
                    else:
                        sumv += cs.pixeldataIn[cs.unif_slice,roix[0]+kx,roiy[0]+ky]
                    # self.pixeldataIn[iz,roix[0]+kx,roiy[0]+ky] = -1000
                    count += 1
        if(count>0):
            cs.skull_avg = sumv/count
        else:
            cs.skull_avg = 0

        # for gui
        cs.unif_rois.append([roix[0]+outerroidim/2,roiy[0]+outerroidim/2, outerroidim/2])
        cs.unif_rois.append([roix[1]+innerroidim/2,roiy[1]+innerroidim/2, innerroidim/2])

        cs.snr_hol = (cs.roiavg[0]+1000)/cs.roisd
        # uniformity
        maxdev = abs(cs.roiavg[0]-cs.roiavg[1])
        for k in  range(2,3):
            maxdev = max(maxdev,abs(cs.roiavg[0]-cs.roiavg[k]))
        cs.unif = -maxdev

        measured = [cs.roiavg[-1],cs.roiavg[0],cs.skull_avg]
        GT = cs.guessScanner.HeadGT
        print("[AnalyseCTHead] x,y",GT,measured)
        cs.linearity = self.PearsonCoef(measured,GT)

        maxdev = measured[0]-GT[0]
        for m,g in zip(measured,GT):
            if(abs(m-g)>abs(maxdev)):
                maxdev = m-g
        cs.maxdev = maxdev

        error = False
        return error

    def AnalyseCTBody(self, cs):
        """
        Calculate uniformity, water#, noise, CT#, shift
        1. Find fantom center shift
        2. Find mean and SD in relevant ROIs
        3. Find mean and SD in skull
        4. Find
        """
        error = True

        # 1. find center shift
        if cs.dicomMode == wadwrapper_lib.stMode2D:
            error,cs.shiftxypx = self.FindCenterShift2D(cs,cs.pixeldataIn,matter=lit.stAir)
        else:
            error,cs.shiftxypx = self.FindCenterShift2D(cs,cs.pixeldataIn[cs.unif_slice],matter=lit.stAir)
        if error:
            print("[AnalyseCTBody] cannot findcentershift")
            return error

        # 2. find mean and SD in relevant regions
        #   1  2  3
        #
        #5  0  4
        #
        if cs.dicomMode == wadwrapper_lib.stMode2D:
            datadiam = cs.dcmInfile.ReconstructionDiameter # data reconstruction diameter
        else:
            datadiam = cs.dcmInfile.info.ReconstructionDiameter # data reconstruction diameter
        midx     = int(self.phantommm2pix(cs,datadiam/2.)+.5)
        stepx    = int(self.phantommm2pix(cs,120.)+.5)
        roidimIEC   = int(self.phantommm2pix(cs,30.)+.5) # 25 for body // IEC say at least 10% of diam (178mm)
        roidimMAT   = int(self.phantommm2pix(cs,18.)+.5) # 2non official, like for uniformity and outside
        sddimIEC    = int(3.25*roidimIEC)#IEC say at least 40% of diam (178mm) and may not overlap with other structures
        hdiagstep = int(self.phantommm2pix(cs,85.)+.5)
        diagstep = int(self.phantommm2pix(cs,cs.guessScanner.BodyAirDistmm)+.5) # distance to Air ROI 

        roidim = []
        roidim.append(roidimIEC)
        roidim.append(roidimIEC)
        roidim.append(roidimIEC)
        roidim.append(roidimMAT)
        roidim.append(roidimMAT)
        roidim.append(roidimMAT)
        roidim.append(sddimIEC)
        roix = []
        roiy = []
        roix.append(int(cs.shiftxypx[0]+midx-int(roidim[0]/2))) # center
        roix.append(roix[0])            # 12 o'clock
        roix.append(roix[0]+hdiagstep)  # 1.3 o'clock
        roix.append(int(cs.shiftxypx[0]+midx-int(roidim[3]/2)+diagstep)) # outside
        roiy.append(int(cs.shiftxypx[1]+midx-int(roidim[0]/2))) # center
        roiy.append(roiy[0]-stepx)            # 12 o'clock
        roiy.append(roiy[0]-hdiagstep)        # 1.3 o'clock
        roiy.append(int(cs.shiftxypx[1]+midx-int(roidim[3]/2)-diagstep)) # outside

        # Find Teflon plug
        stepm  = int(self.phantommm2pix(cs,82.)+.5)
        matdim = int(self.phantommm2pix(cs,55.)+.5)
        roix.append(int(cs.shiftxypx[0]+midx-int(roidim[4]/2)))
        roiy.append(int(cs.shiftxypx[1]+midx-int(roidim[4]/2)))
        xstart = int(roix[4]+stepm-matdim/2)
        xend   = xstart+matdim
        ystart = int(roiy[4]-matdim/2)
        yend   = ystart+matdim
        if cs.dicomMode == wadwrapper_lib.stMode2D:
            plugimage = cs.pixeldataIn[xstart:xend,ystart:yend]
        else:
            plugimage = cs.pixeldataIn[cs.unif_slice,xstart:xend,ystart:yend]
        error,shiftxyTeflon = self.FindCenterShift2D(cs,plugimage,matter=lit.stTeflon) ## FIXME: moet toch pixeldata toestaan en shiftxy uitspugen
        print("[AnalyseCTBody] Teflon shift=",shiftxyTeflon)
        roix[4] = int(shiftxyTeflon[0]+(xstart+xend)/2-int(roidim[4]/2)+1) #  3 o'clock material
        roiy[4] = int(shiftxyTeflon[1]+(ystart+yend)/2-int(roidim[4]/2)+1)
        if(cs.verbose):
            plt.figure()
            plt.imshow(plugimage)
            plt.title("Teflon")
            cs.hasmadeplots = True

        if cs.guessScanner.BodyGT[2] >-900.: # <-900 means no water plug, so use air outside.
            # Find Water plug
            roix.append(int(cs.shiftxypx[0]+midx-int(roidim[5]/2)))
            roiy.append(int(cs.shiftxypx[1]+midx-int(roidim[5]/2)))
            xstart = int(roix[5]-stepm-matdim/2)
            xend   = xstart+matdim
            ystart = int(roiy[5]-matdim/2)
            yend   = ystart+matdim
            if cs.dicomMode == wadwrapper_lib.stMode2D:
                plugimage = cs.pixeldataIn[xstart:xend,ystart:yend]
            else:
                plugimage = cs.pixeldataIn[cs.unif_slice,xstart:xend,ystart:yend]
            error,shiftxyWater = self.FindCenterShift2D(cs,plugimage,matter=lit.stWater)
            print("[AnalyseCTBody] Water shift=",shiftxyWater)
            roix[5] = int(shiftxyWater[0]+(xstart+xend)/2-int(roidim[5]/2)+1) #  9 o'clock material
            roiy[5] = int(shiftxyWater[1]+(ystart+yend)/2-int(roidim[5]/2)+1)
            if(cs.verbose):
                plt.figure()
                plt.imshow(plugimage)
                plt.title("Water")
                cs.hasmadeplots = True

        # sd roi
        roix.append(int(cs.shiftxypx[0]+midx-int(roidim[6]/2))) # center
        roiy.append(int(cs.shiftxypx[1]+midx-int(roidim[6]/2))) # center

        # first avg
        cs.roiavg = []
        for kk in range(0,len(roix)-1):
#        for kk in range(0,1):
            roirad = roidim[kk]/2.
            sumv = 0.;
            count = 0
            for ky in range(0,roidim[kk]):
                for kx in range(0,roidim[kk]):
                    if( (kx-roirad)**2+(ky-roirad)**2<roirad**2 ):
                        if cs.dicomMode == wadwrapper_lib.stMode2D:
                            sumv += cs.pixeldataIn[roix[kk]+kx,roiy[kk]+ky]
                        #    self.pixeldataIn[roix[kk]+kx,roiy[kk]+ky] = -1000
                        else:
                            sumv += cs.pixeldataIn[cs.unif_slice,roix[kk]+kx,roiy[kk]+ky]
                        # self.pixeldataIn[iz,roix[kk]+kx,roiy[kk]+ky] = -1000
                        count += 1
            if(count>0):
                cs.roiavg.append(sumv/count)
            else:
                cs.roiavg.append(0)
            # for gui
            cs.unif_rois.append([roix[kk]+roidim[kk]/2,roiy[kk]+roidim[kk]/2, roidim[kk]/2])

        # now SD
        cs.roisd = 0.;
        sdrad = roidim[-1]/2.
        sumv = np.float32(0.);
        sumv2 = np.float32(0.)
        count = 0
        np.seterr(over='raise')
        for ky in range(0,roidim[-1]):
            for kx in range(0,roidim[-1]):
                if( (kx-sdrad)**2+(ky-sdrad)**2<sdrad**2 ):
                    if cs.dicomMode == wadwrapper_lib.stMode2D:
                        val = cs.pixeldataIn[roix[-1]+kx,roiy[-1]+ky]
                    else:
                        val = cs.pixeldataIn[cs.unif_slice,roix[-1]+kx,roiy[-1]+ky]
                    sumv += val
                    try:
                        sumv2 += val*val
                    except:
                        print("ERROR!",type(val),type(sumv),type(sumv2),val,sumv,sumv2)
                    #  self.pixeldataIn[iz,roix[kk]+kx,roiy[kk]+ky] = -1000
                    count += 1
        if(count>1):
            cs.roisd = (count*sumv2-sumv*sumv)/count
            cs.roisd = np.sqrt(cs.roisd/(count-1.))

        # for gui
        cs.unif_rois.append([roix[-1]+roidim[-1]/2,roiy[-1]+roidim[-1]/2, roidim[-1]/2])

        cs.snr_hol = (cs.roiavg[0]+1000)/cs.roisd

        # uniformity
        maxdev = abs(cs.roiavg[0]-cs.roiavg[1])
        for k in  range(2,3):
            maxdev = max(maxdev,abs(cs.roiavg[0]-cs.roiavg[k]))
        cs.unif = -maxdev

        # linearity
        if cs.guessScanner.BodyGT[2] <-900.: # <-900 means no water plug, so use air outside.
            measured = [cs.roiavg[0],cs.roiavg[4],cs.roiavg[3]]# use air
        else:
            measured = [cs.roiavg[0],cs.roiavg[4],cs.roiavg[5]]
        GT = cs.guessScanner.BodyGT
        print("[AnalyseCTBody] x,y",GT,measured)
        cs.linearity = self.PearsonCoef(measured,GT)
        maxdev = measured[0]-GT[0]
        for m,g in zip(measured,GT):
            if(abs(m-g)>abs(maxdev)):
                maxdev = m-g
        cs.maxdev = maxdev

        error = False
        return error

#----------------------------------------------------------------------
    def PearsonCoef(self,yarr,xarr):
        r1_value = 0;
        nonidentical = True
        ssxm, ssxym, ssyxm, ssym = np.cov(xarr,yarr, bias=1).flat
        if(ssxm == 0):
            nonidentical = False
        if(nonidentical):
            slope, intercept, r1_value, p_value, std_err = stats.linregress(xarr,yarr)
        return r1_value**2

    def FindCenterShift2D(self,cs,pixeldata,matter):
        error = True

        shiftxypx = [0,0]
        """
        Needs 2d input pixeldata. If 3D, then input pixeldata[slice]
        Concept:
            1. blur with object dependend scale
        """
        if(np.shape(np.shape(pixeldata))[0]!=2):
            print("[FindCenterShift2D] Error, called with non-2D data!")
            return error

        wid = np.shape(pixeldata)[0] ## width/height in pixels
        hei = np.shape(pixeldata)[1]

        dscale = 7.0
        thresh = -500
        lowmode = True
        if(matter == lit.stTungsten):
            error = False
            dscale = 1.
            thresh = 3
        if(matter == lit.stTeflon):
            error = False
            thresh = 250
        if(matter == lit.stWater):
            error = False
            thresh = 50
            lowmode = False
        if(matter == lit.stAir):
            error = False

        # 1. blur object
        blurIm = scipy.ndimage.gaussian_filter(pixeldata, sigma=dscale)
        # 2. only look for voxels below/above a threshold (eg. look for air)
        arrayLow = []
        for iy in range(0,hei):
            arrayLow.append(0)
            for ix in range(0,wid):
                if(lowmode):
                    if(blurIm[ix,iy]<thresh):
                        arrayLow[iy] += 1
                else:
                    if(blurIm[ix,iy]>thresh):
                        arrayLow[iy] += 1

        if(cs.verbose):
            plt.figure()
            plt.plot(arrayLow)
            plt.title("vertical "+matter)
            cs.hasmadeplots = True

        # 2.1 find left first pos without voxels below threshLow
        minLowId = 0
        minLow = arrayLow[minLowId]
        for iy in range(0,hei):
            if(arrayLow[iy]<minLow):
                minLowId = iy
                minLow = arrayLow[minLowId]
        shiftxypx[1] = minLowId
        if(cs.verbose):
            print("[FindCenterShift] vertical "+matter,minLowId)
        # 2.2 find right first pos without voxels below threshLow
        minLowId = hei-1
        minLow = arrayLow[minLowId]
        for iy in reversed(range(0,hei)):
            if(arrayLow[iy]<minLow):
                minLowId = iy
                minLow = arrayLow[minLowId]
        # 2.3 mid is halfway left and right pos
        shiftxypx[1] =int((shiftxypx[1]+minLowId-(hei-1))/2)
        if(cs.verbose):
            print("[FindCenterShift] vertical "+matter,minLowId,(hei-1)/2,shiftxypx[1])
        # repeat for horizontal
        arrayLow = []
        for ix in range(0,wid):
            arrayLow.append(0)
            for iy in range(0,hei):
                if(lowmode):
                    if(blurIm[ix,iy]<thresh):
                        arrayLow[ix] += 1
                else:
                    if(blurIm[ix,iy]>thresh):
                        arrayLow[ix] += 1

        if(cs.verbose):
            plt.figure()
            plt.plot(arrayLow)
            plt.title("horizontal "+matter)
            cs.hasmadeplots = True

        # 2.1 find left first pos without voxels below threshLow
        minLowId = 0
        minLow = arrayLow[minLowId]
        for iy in range(0,wid):
            if(arrayLow[iy]<minLow):
                minLowId = iy
                minLow = arrayLow[minLowId]
        shiftxypx[0] = minLowId
        if(cs.verbose):
            print("[FindCenterShift] horizontal "+matter,minLowId)

        # 2.2 find right first pos without voxels below threshLow
        minLowId = wid-1
        minLow = arrayLow[minLowId]
        for iy in reversed(range(0,wid)):
            if(arrayLow[iy]<minLow):
                minLowId = iy
                minLow = arrayLow[minLowId]
        # 2.3 mid is halfway left and right pos
        shiftxypx[0] =int((shiftxypx[0]+minLowId-(wid-1))/2)
        if(cs.verbose):
            print("[FindCenterShift] horizontal "+matter,minLowId,(wid-1)/2,shiftxypx[0])

        error = False
        return error,shiftxypx

#----------------------------------------------------------------------
    def DICOMInfo(self,cs,info='dicom'):
        # Different from ImageJ version; tags "0008","0104" and "0054","0220"
        #  appear to be part of sequences. This gives problems (cannot be found
        #  or returning whole sequence blocks)
        # Possibly this can be solved by using if(type(value) == type(dicom.sequence.Sequence()))
        #  but I don't see the relevance of these tags anymore, so set them to NO

        if(info == "dicom"):
            dicomfields = [
                ["0008,0022", "Acquisition Date"],
                ["0008,0032", "Acquisition Time"],
                ["0008,0060", "Modality"],
                ["0008,0070", "Manufacturer"],
                ["0008,1010", "Station Name"],
                ["0008,103e", "Series Description"],
                ["0008,1010", "Station Name"],
                ["0018,0022", "Scan Options"], # Philips
                ["0018,0050", "Slice Thickness"],
                ["0018,0060", "kVp"],
                ["0018,0088", "Spacing Between Slices"], # Philips
                ["0018,0090", "Data Collection Diameter"],
                ["0018,1020", "Software Versions(s)"],
                ["0018,1030", "Protocol Name"],
                ["0018,1100", "Reconstruction Diameter"],
                ["0018,1120", "Gantry/Detector Tilt"],
                ["0018,1130", "Table Height"],
                ["0018,1140", "Rotation Direction"],
                ["0018,1143", "Scan Arc"], # noPhilips noSiemens
                ["0018,1150", "Exposure Time ms"], #Siemens
                ["0018,1151", "X-ray Tube Current"],
                ["0018,1152", "Exposure mAs"], # mA*tRot/pitch; tRot=exposure time
                ["0018,9345", "CTDIvol"],
                ["0018,1160", "Filter Type"],
                ["0018,1210", "Convolution Kernel"],
                ["0018,5100", "Patient Position"],
                ["0020,0013", "Image Number"],
                ["0020,1041", "Slice Location"],
                ["0028,0030", "Pixel Spacing"],
                ["01F1,1027", "Rotation Time"], # Philips
                ["01F1,104B", "Collimation"], # Philips
                ["01F1,104E", "Protocol"] ] # Philips

        elif(info == "idose"):
            dicomfields = [
                #"0018,9323", "Recon",
                ["01F7,109B", "iDose"] ]

        elif(info == "id"):
            dicomfields = [
                ["0018,1030", "ProtocolName"],
                ["0008,103e", "SeriesDescription"],
                ["0008,0022", "AcquisitionDate"],
                ["0008,0032", "AcquisitionTime"]
            ]

        results = []
        for df in dicomfields:
            key = df[0]
            value = ""
            if(key=="0018,0088"): #DICOM spacing
                value = self.readDICOMspacing(cs)
            else:
                try:
                    value = self.readDICOMtag(cs,key)
                except:
                    value = ""
            if(key=="0018,1020"):
                value = "'"+value
            results.append( (df[1],value) )

        return results

    def saveAnnotatedImage(self, cs, fname):
        # make a palette, mapping intensities to greyscale
        pal = np.arange(0,256,1,dtype=np.uint8)[:,np.newaxis] * \
            np.ones((3,),dtype=np.uint8)[np.newaxis,:]
        # but reserve the first for red for markings
        pal[0] = [255,0,0]

        rectrois = []
        polyrois = []
        circlerois = []

        # convert to 8-bit palette mapped image with lowest palette value used = 1
        # first the base image
        if cs.dicomMode == wadwrapper_lib.stMode2D:
            im = scipy.misc.toimage(cs.pixeldataIn.transpose(), low=1, pal=pal) # MODULE EXPECTS PYQTGRAPH DATA: X AND Y ARE TRANSPOSED!
        else:
            im = scipy.misc.toimage((cs.pixeldataIn[cs.unif_slice]).transpose(),low=1,pal=pal) # MODULE EXPECTS PYQTGRAPH DATA: X AND Y ARE TRANSPOSED!

        # add all rois
        for r in cs.unif_rois: # uniformity
            circlerois.append(r)

        # now draw all rois in reserved color
        draw = ImageDraw.Draw(im)
        for r in polyrois:
            roi =[]
            for x,y in r:
                roi.append( (int(x+.5),int(y+.5)))
            draw.polygon(roi,outline=0)

        for r in rectrois:
            #draw.rectangle(r,outline=0)
            self.drawThickRectangle(draw, r, 0, 3)

        # now draw all cirlerois in reserved color
        for x,y,r in circlerois: # low contrast elements
            draw.ellipse((x-r,y-r,x+r,y+r), outline=0)
        del draw

        # convert to RGB for JPG, cause JPG doesn't do PALETTE and PNG is much larger
        im = im.convert("RGB")

        imsi = im.size
        if max(imsi)>2048:
            ratio = 2048./max(imsi)
            im = im.resize( (int(imsi[0]*ratio+.5), int(imsi[1]*ratio+.5)),Image.ANTIALIAS)
        im.save(fname)

#----------------------------------------------------------------------
