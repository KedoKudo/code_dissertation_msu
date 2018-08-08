#!/usr/bin/env python

import numpy as np
import h5py as h5
import xml.etree.cElementTree as ET
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from cyxtal import slip_systems
from cyxtal import bravais2cartesian
from cyxtal.cxtallite import Eulers


def find_element(h5ImageName, xmlFileName, namespace):
    """find the element in the indexation file w.r.t the input image"""
    ns = namespace
    xmlfilen = xmlFileName
    h5Im = h5ImageName.split('/')[-1]
    
    # use celementtree for fast data retriving
    tree = ET.parse(xmlfilen)
    root = tree.getroot()
    ei = -1  # the child id for element containing given image  
    
    # first, try to find the element contains the image
    for i in range(len(root)):
        step = root[i]

        tmp_str = 'step:detector/step:inputImage'
        imgn = step.find(tmp_str, ns)
        if imgn is None:
            continue
        else:
            imgn = imgn.text.split('/')[-1]
            if imgn == h5Im:
                ei = i 
                break
    
    # if no element contains this image, 
    # which means no indexation results, 
    # return None
    if ei < 0:
        return None
    else:
        step = root[ei]
        # check if indexation are successfully computed
        tmp_str = 'step:indexing/step:pattern/step:recip_lattice/step:astar'
        astar = step.find(tmp_str, ns)
        if astar is None:
            return None
        
        # get motor/wire position
        xsample = step.find('step:Xsample', ns).text
        ysample = step.find('step:Ysample', ns).text
        zsample = step.find('step:Zsample', ns).text
        depth = step.find('step:depth', ns).text
        # get peak position on CCD
        xpix = step.find('step:detector/step:peaksXY/step:Xpixel', 
                         ns).text
        ypix = step.find('step:detector/step:peaksXY/step:Ypixel', 
                         ns).text
        # get reciprocal lattice vectors
        astar_str = 'step:indexing/step:pattern/step:recip_lattice/step:astar'
        bstar_str = 'step:indexing/step:pattern/step:recip_lattice/step:bstar'
        cstar_str = 'step:indexing/step:pattern/step:recip_lattice/step:cstar'
        astar = step.find(astar_str, ns).text
        bstar = step.find(bstar_str, ns).text
        cstar = step.find(cstar_str, ns).text
        # get diffraction vectors
        qx = step.find('step:detector/step:peaksXY/step:Qx', ns).text
        qy = step.find('step:detector/step:peaksXY/step:Qy', ns).text
        qz = step.find('step:detector/step:peaksXY/step:Qz', ns).text
        # get index results (hkl)
        h = step.find('step:indexing/step:pattern/step:hkl_s/step:h',
                      ns).text
        k = step.find('step:indexing/step:pattern/step:hkl_s/step:k',
                      ns).text
        l = step.find('step:indexing/step:pattern/step:hkl_s/step:l',
                      ns).text
        
        # packing data
        rst = {'xsample': float(xsample),
               'ysample': float(ysample),
               'zsample': float(zsample),
               'depth': float(depth),
               'astar': map(float, astar.split()),
               'bstar': map(float, bstar.split()),
               'cstar': map(float, cstar.split()),
               'peakX': map(float, xpix.split()),
               'peakY': map(float, ypix.split()),
               'h': map(float, h.split()),
               'k': map(float, k.split()),
               'l': map(float, l.split()),
               'qx': map(float, qx.split()),
               'qy': map(float, qy.split()),
               'qz': map(float, qz.split()),}
        return rst


def plt_pattern(DAXM_Data):
    """plot streak dials based on DAXM indexation results"""
    xsample = DAXM_Data['xsample']
    ysample = DAXM_Data['ysample']
    zsample = DAXM_Data['zsample']
    depth = DAXM_Data['depth']
    
    peakx = np.array(DAXM_Data['peakX'])
    peaky = np.array(DAXM_Data['peakY'])
    
    astar = np.array(DAXM_Data['astar'])
    bstar = np.array(DAXM_Data['bstar'])
    cstar = np.array(DAXM_Data['cstar'])
    g_xtal2APS = gFromReciprocalLattice(astar, bstar, cstar)
    
    # get reciprocal lattice
    rl = np.column_stack((astar, bstar, cstar))
    
    h = np.array(DAXM_Data['h'])
    k = np.array(DAXM_Data['k'])
    l = np.array(DAXM_Data['l'])
    hkl_crystal = np.column_stack((h, k, l))
    
    qx = np.array(DAXM_Data['qx'])
    qy = np.array(DAXM_Data['qy'])
    qz = np.array(DAXM_Data['qz'])
    q_aps = np.column_stack((qx, qy, qz))
    
    # data pruning, remove non-indexed diffraction spots
    threshold = 1e-4
    new_px = []
    new_py = []
    new_qs = []
    new_hkl = []
    
    print "original hkl"
    print len(hkl_crystal), len(q_aps), len(peakx)
    # HKL is the indexation results, it will always be a subset of
    # the number of peaks found in the diffraction pattern. So the 
    # pruning part is basically just to make sure that the correct
    # peak (q and x,y) is associated with each hkl (indexation)
    
    for i, hkl in enumerate(hkl_crystal):
        q_calc = np.dot(rl, hkl)
        q_calc = q_calc / np.linalg.norm(q_calc)
        # compute the difference from each q
        diff = [1.0-abs(np.dot(q_mea, q_calc)) for q_mea in q_aps]
        # find the min diff
        if min(diff) > threshold:
            msg = "threshold too tight! "
            msg += "Cannot find q vectors to match indexation"
            raise ValueError(msg)
        else:
            minIdx = np.argmin(diff)
            new_px.append(peakx[minIdx])
            new_py.append(peaky[minIdx])
            new_qs.append(q_aps[minIdx])
            new_hkl.append(hkl)
                
    # use the pruned data from this point
    peakx = np.array(new_px)
    peaky = np.array(new_py)

    
    # covert the plane normal to aps coordinate system
    # then convert to img coordinate system.
    # see
    #   https://www1.aps.anl.gov/files/download/SECTORS33-34/coordinates-PE-system.pdf
    hkl_aps = [np.dot(g_xtal2APS, me) for me in hkl_crystal]
    peaks = calcPeakPosition(hkl_aps)
    g_aps2img = np.array([[0, 0, 1], 
                          [1, 0, 0], 
                          [0, 1, 0]])
    hkl_img = []
    x_peak = []
    y_peak = []
    for i in xrange(len(hkl_aps)):
        # first calculate theoretical spot 
        peak_pos = peaks[i, :]
        tmp_x, tmp_y, tmp_z = np.dot(g_aps2img, peak_pos)
        x_peak.append(tmp_x/tmp_z)
        y_peak.append(tmp_y/tmp_z)
        # prepare unit HKL for streak analysis
        me = np.array(hkl_aps[i])
        # the algorithem needs normalized vector
        me = me/np.linalg.norm(me)          
        hkl_img.append(np.dot(g_aps2img, me))
    hkl_img = np.array(hkl_img)
    
    
    # plot scatter plot 
    plt.figure(figsize=(15,5))
    
    # peak position from DAXM data
    plt.subplot(131)
    plt.scatter(peakx, peaky, color='r', marker="s")
    ax = plt.gca()
    for i in xrange(len(peakx)):
        ax.annotate("{}_{}".format(i, new_hkl[i]), (peakx[i],peaky[i]))

    # original X-ray pattern
    plt.subplot(132)
    cmap = plt.cm.gray
    cmap.set_under(color='white')
    plt.imshow(np.transpose(imgData.T),
               origin='lower',
               cmap=cmap,
               norm=LogNorm(vmin=1e2, vmax=np.amax(np.amax(imgData))))
    
    # peak position
    plt.subplot(133)
    x_peak = np.array(x_peak)
    y_peak = np.array(y_peak)
    plt.scatter(x_peak, y_peak, color='b', marker="^")
    plt.savefig('test.png')
    
    for i in xrange(len(x_peak)):
        print x_peak[i], y_peak[i], " | ", hkl_crystal[i]
    
    # sanity check, ensure match
    print len(peakx), len(hkl_crystal), len(x_peak)  

    for i in xrange(len(hkl_img)):
        N = hkl_crystal[i]
        xi = [np.cross(N, ti) for ti in t]
        cx = peakx[i]
        cy = peaky[i]
        
        # not weighted by schmid factor
        plt.clf()
        plt.figure(frameon=False, figsize=(5,5))
        xs = []
        ys = []
        ls = []
        for j, me in enumerate(xi):
            tmp = np.dot(g_aps2img, np.dot(g_xtal2APS, me))
            vnorm = np.linalg.norm(tmp)
            tmp = tmp/vnorm  # normalized the 3D vector
            
            x,y,z = tmp  # from crystal to APS to IMG
            norm = np.sqrt(x**2 + y**2)
            ls.append(norm)  # this will be used to scale the chages.
            xs.append(x/norm)
            ys.append(y/norm)  # projection scaling handled through l

        # make the longest project 1, others scales relatively
        xs = np.array(xs)
        ys = np.array(ys)
        ls = np.array(ls)/max(ls)  

        radius_dial = 1.0
        offset_dial = 1.0
        for j, me in enumerate(xi):
            x = xs[j]
            y = ys[j]
            l = ls[j]*radius_dial/2.
            
            angle = np.arctan2(y,x)
            radius = [1,2,3,3,4,4,4,4][j//3]*radius_dial+offset_dial
            
            plt.plot( ((radius-l)*np.cos(angle),
                       (radius+l)*np.cos(angle)
                      ), 
                      ((radius-l)*np.sin(angle),
                       (radius+l)*np.sin(angle)
                      ),
                      lnstyle[j], color=clrs[j], linewidth=lw )

            plt.plot( ((radius-l)*np.cos(angle+np.pi),
                       (radius+l)*np.cos(angle+np.pi)
                       ), 
                      ((radius-l)*np.sin(angle+np.pi),
                       (radius+l)*np.sin(angle+np.pi)
                      ),
                      lnstyle[j], color=clrs[j], linewidth=lw )

        ax = plt.gca()

        for family in np.arange(4):
            ax.add_artist(plt.Circle((0, 0), 
                                    (family+1)*radius_dial+offset_dial, 
                                    fill=False, 
                                    edgecolor=(0.9,0.9,0.9), 
                                    alpha=1, 
                                    linewidth=lw/2.
                                   )
                        )
        
        plt.axis('off')
        h,k,l = map(int, new_hkl[i])
        plt.xlim([-6,6])
        plt.ylim([-6,6])
        plt.tight_layout()
        plt.savefig("streak{}_[{}_{}_{}].pdf".format(i,h,k,l), 
                    dpi=240, 
                    transparent=True,
                    pad_inches=0)
        
        plt.clf()
        plt.figure(frameon=False, figsize=(5,5))
        for j, me in enumerate(xi):
            x = xs[j]
            y = ys[j]
            l = ls[j]*radius_dial/2.
            angle = np.arctan2(y,x)
            radius = [1,2,3,3,4,4,4,4][j//3]*radius_dial+offset_dial
            
            plt.plot( ((radius-l)*np.cos(angle),
                       (radius+l)*np.cos(angle)
                      ), 
                      ((radius-l)*np.sin(angle),
                       (radius+l)*np.sin(angle)
                      ),
                      lnstyle[j], alpha=ms[j], 
                      color=clrs[j], linewidth=lw 
                    )

            plt.plot( ((radius-l)*np.cos(angle+np.pi),
                       (radius+l)*np.cos(angle+np.pi)
                      ), 
                      ((radius-l)*np.sin(angle+np.pi),
                       (radius+l)*np.sin(angle+np.pi)
                      ),
                      lnstyle[j], alpha=ms[j], 
                      color=clrs[j], linewidth=lw 
                    )

        ax = plt.gca()

        for family in np.arange(4):
            ax.add_artist(plt.Circle((0, 0),
                                    (family+1)*radius_dial+offset_dial,
                                     fill=False, 
                                     edgecolor=(0.9,0.9,0.9), 
                                     alpha=1, 
                                     linewidth=lw/2.
                                    )
                         )
        
        plt.axis('off')
        h,k,l = map(int, new_hkl[i])
        plt.xlim([-6,6])
        plt.ylim([-6,6])
        plt.tight_layout()
        plt.savefig("streakwgt{}_[{}_{}_{}].pdf".format(i,h,k,l), 
                    dpi=240, 
                    transparent=True,
                    pad_inches=0)



# process single voxel data
xmlFileName = "Ti525_H1H2.xml"
h5ImageName = "H2_images/Ti525_H1H2_38077_59.h5"
ns = {'step':'http://sector34.xray.aps.anl.gov/34ide:indexResult'}

h5f = h5.File(h5ImageName, 'r')
h5fDst = h5f['entry1/data/data']
imgData = np.zeros(h5fDst.shape)
h5fDst.read_direct(imgData)

# save the diffraction pattern
plt.figure(frameon=False, figsize=(5,5))
cmap = plt.cm.gray_r
cmap.set_under(color='white')
plt.imshow(np.transpose(imgData.T),
           origin='lower',
           vmin=1e2,
           cmap=cmap)
plt.axis('off')

plt.savefig('pattern.png', 
            dpi=240, 
            transparent=True,
            bbox_inches='tight',
            pad_inches=0)
plt.clf()    

daxm_data = find_element(h5ImageName, xmlFileName, ns)

plt_pattern(daxm_data)

