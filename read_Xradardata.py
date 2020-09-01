import os
import bz2
import pandas as pd
import numpy as np


variable_encode = {1: "dbt", 2: "dbz", 7: "zdr", 9: "cc", 10: "phidp", 11: "kdp"}

radarDir = "data/Xband"
radarDataFile = "BJXSY.20170822.080000.AR2"
fid = open(os.path.join(radarDir, radarDataFile), "rb")

# Shift value for file pointers of different block
# ---generic header block pointer (len=32)
pt_common_block = 0
# ---site configuration information block pointer (len=128)
pt_site_block = 0 + 32
# ---task configuration information block pointer (len=256)
pt_task_block = 0 + 32 + 128
# ---1st angle of elevation block pointer
# ------The shift value of N-th angle of elevation is: 416 + (N-1) * 256
pt_1st_ele = 0 + 32 + 128 + 256

f = pd.DataFrame(columns=['siteinfo', 'taskinfo', 'eleinfo', 'radinfo', 'dbt', 'dbz', 'zdr', 'cc', 'phidp', 'kdp'])

# ---read site configuration information
fid.seek(pt_site_block, 0)
# ------site code
f['siteinfo'].loc[0] = {'code': ''.join([chr(item) for item in np.fromfile(file=fid, dtype=np.int8, count=8)])}
# ------site name
f['siteinfo'].loc[1] = {'name': ''.join([chr(item) for item in np.fromfile(fid, np.int8, 32)])}
# ------latitude
f['siteinfo'].loc[2] = {'lat': np.fromfile(fid, np.float32, 1)[0]}
# ------longitude
f['siteinfo'].loc[3] = {'lon': np.fromfile(fid, np.float32, 1)[0]}
# ------antenna height
f['siteinfo'].loc[4] = {'atennaasl': np.fromfile(fid, np.int32, 1)[0]}
# ------ground height
f['siteinfo'].loc[5] = {'baseasl': np.fromfile(fid, np.int32, 1)[0]}
# ------frequency
f['siteinfo'].loc[6] = {'freq': np.fromfile(fid, np.float32, 1)[0]}
# ------beam width (Horizontal)
f['siteinfo'].loc[7] = {'beamhwidth': np.fromfile(fid, np.float32, 1)[0]}
# ------beam width (Vertical)
f['siteinfo'].loc[8] = {'beamvwidth': np.fromfile(fid, np.float32, 1)[0]}

# ---read task configuration information
fid.seek(pt_task_block, 0)
# ------task name
f['taskinfo'].loc[0] = {'name': ''.join([chr(item) for item in np.fromfile(fid, np.int8, 32)])}
# ------task description
f['taskinfo'].loc[1] = {'description': ''.join([chr(item) for item in np.fromfile(fid, np.int8, 128)])}
# ------polarization type
f['taskinfo'].loc[2] = {'polmode': np.fromfile(fid, np.int32, 1)[0]}
# ------scan type: 1 for Plan Position Indicator (PPI), 2 for Range Height Indicator (RHI)
f['taskinfo'].loc[3] = {'scantype': np.fromfile(fid, np.int32, 1)[0]}
# ------pulse width
f['taskinfo'].loc[4] = {'pulsewidth': np.fromfile(fid, np.int32, 1)[0]}
# ------!!! scan start time (UTC, start from 1970/01/01 00:00)
f['taskinfo'].loc[5] = {'startime': np.fromfile(fid, np.int32, 1)[0]}
# ------!!! cut number
f['taskinfo'].loc[6] = {'cutnum': np.fromfile(fid, np.int32, 1)[0]}
# ------horizontal noise
f['taskinfo'].loc[7] = {'hnoise': np.fromfile(fid, np.float32, 1)[0]}
# ------vertical noise
f['taskinfo'].loc[8] = {'vnoise': np.fromfile(fid, np.float32, 1)[0]}
# ------horizontal calibration
f['taskinfo'].loc[9] = {'hsyscal': np.fromfile(fid, np.float32, 1)[0]}
# ------vertical calibration
f['taskinfo'].loc[10] = {'vsyscal': np.fromfile(fid, np.float32, 1)[0]}
# ------horizontal noise temperature
f['taskinfo'].loc[11] = {'hte': np.fromfile(fid, np.float32, 1)[0]}
# ------vertical noise temperature
f['taskinfo'].loc[12] = {'vte': np.fromfile(fid, np.float32, 1)[0]}
# ------ZDR calibration
f['taskinfo'].loc[13] = {'zdrbias': np.fromfile(fid, np.float32, 1)[0]}
# ------PhiDP calibration
f['taskinfo'].loc[14] = {'phasebias': np.fromfile(fid, np.float32, 1)[0]}
# ------LDR calibration
f['taskinfo'].loc[15] = {'ldrbias': np.fromfile(fid, np.float32, 1)[0]}

# ---read angle of elevation block
for ct in range(1, int(f['taskinfo'].loc[6]['cutnum'])+1):
	fid.seek(pt_1st_ele + (ct-1) * 256, 0)
	info_dict = {}
	# ------process mode
	info_dict['mode'] = np.fromfile(fid, np.int32, 1)[0]
	# ------wave form
	info_dict['waveform'] = np.fromfile(fid, np.int32, 1)[0]
	# ------pulse repetition frequency 1 (PRF1)
	info_dict['prf1'] = np.fromfile(fid,np.float32, 1)[0]
	# ------pulse repetition frequency 2 (PRF2)
	info_dict['prf2'] = np.fromfile(fid, np.int32, 1)[0]
	# ------de-aliasing mode
	info_dict['unfoldmode'] = np.fromfile(fid, np.int32, 1)[0]
	# ------azimuth for RHI mode
	info_dict['azi'] = np.fromfile(fid, np.float32, 1)[0]
	# ------elevation for PPI mode
	info_dict['ele'] = np.fromfile(fid, np.float32, 1)[0]
	# ------start angle, i.e., start azimuth for PPI mode or highest elevation for RHI mode
	info_dict['startangle'] = np.fromfile(fid, np.int32, 1)[0]
	# ------end angle, i.e., end azimuth for PPI mode or lowest elevation for RHI mode
	info_dict['endangle'] = np.fromfile(fid, np.float32, 1)[0]
	# ------angular resolution (only for PPI mode)
	info_dict['angleres'] = np.fromfile(fid, np.float32, 1)[0]
	# ------scan speed
	info_dict['scanspeed'] = np.fromfile(fid, np.float32, 1)[0]
	# ------log resolution
	info_dict['logres'] = np.fromfile(fid, np.float32, 1)[0]
	# ------Doppler Resolution
	info_dict['dopres'] = np.fromfile(fid, np.float32, 1)[0]
	# ------Maximum Range corresponding to PRF1
	info_dict['maxrange1'] = np.fromfile(fid, np.float32, 1)[0]
	# ------Maximum Range corresponding to PRF2
	info_dict['maxrange2'] = np.fromfile(fid, np.int32, 1)[0]
	# ------start range
	info_dict['startrange'] = np.fromfile(fid, np.int32, 1)[0]
	# ------number of samples corresponding to PRF1
	info_dict['samplenum1'] = np.fromfile(fid, np.int32, 1)[0]
	# ------number of samples corresponding to PRF2
	info_dict['samplenum2'] = np.fromfile(fid, np.int32, 1)[0]
	# ------phase mode
	info_dict['phasemode'] = np.fromfile(fid, np.int32, 1)[0]
	# ------atmosphere loss
	info_dict['atmosloss'] = np.fromfile(fid, np.int32, 1)[0]
	# ------Nyquist speed
	info_dict['vmax'] = np.fromfile(fid, np.int32, 1)[0]
	# ------moments mask
	info_dict['mask'] = np.fromfile(fid, np.int32, 1)[0]
	# ------moments size mask
	info_dict['masksize'] = np.fromfile(fid, np.float32, 1)[0]
	info_dict['datasizemask'] = [mas for mas in np.fromfile(fid, np.float64, 64)]
	# ------misc filter mask
	info_dict['filtermask'] = np.fromfile(fid, np.int32, 1)[0]
	# ------SQI threshold
	info_dict['sqi'] = np.fromfile(fid, np.float32, 1)[0]
	# ------SIG threshold
	info_dict['sig'] = np.fromfile(fid, np.float32, 1)[0]
	# ------CSR threshold
	info_dict['csr'] = np.fromfile(fid, np.float32, 1)[0]
	# ------LOG threshold
	info_dict['log'] = np.fromfile(fid, np.float32, 1)[0]
	# ------CPA threshold
	info_dict['cpa'] = np.fromfile(fid, np.float32, 1)[0]
	# ------PMI threshold
	info_dict['pmi'] = np.fromfile(fid, np.float32, 1)[0]
	# ------reserved threshold
	info_dict['threshold'] = np.fromfile(fid, np.int8, 8)[0]
	# ------dBT threshold
	info_dict['dbtmask'] = np.fromfile(fid, np.int32, 1)[0]
	# ------dBZ mask
	info_dict['dbzmask'] = np.fromfile(fid, np.int32, 1)[0]
	# ------velocity mask
	info_dict['vmask'] = np.fromfile(fid, np.int32, 1)[0]
	# ------spectrum width mask
	info_dict['wmask'] = np.fromfile(fid, np.int32, 1)[0]
	# ------DP mask
	info_dict['zdrmask'] = np.fromfile(fid, np.int32, 1)[0]
	# ------mask reserved
	info_dict['maskreserved'] = np.fromfile(fid, np.int8, 12)
	# ------flag for scan synchronization
	info_dict['scansync'] = np.fromfile(fid, np.int32, 1)[0]
	# ------scan direction
	info_dict['scandirection'] = np.fromfile(fid, np.int32, 1)[0]
	# ------ground clutter classifier type
	info_dict['cmap'] = np.fromfile(fid, np.int16, 1)[0]
	# ------ground clutter filter type
	info_dict['cfiltertype'] = np.fromfile(fid, np.int16, 1)[0]
	# ------ground clutter filter notch width
	info_dict['cnotchwidth'] = np.fromfile(fid, np.int16, 1)[0]
	# ------ground clutter filter window
	info_dict['cfilterwin'] = np.fromfile(fid, np.int16, 1)[0]
	# ------reserved
	info_dict['twin'] = np.fromfile(fid, np.int8, 1)[0]
	f['eleinfo'].loc[ct-1] = {str(ct): info_dict}

# ---read radial data block
pt_radial_block = pt_1st_ele + int(f['taskinfo'].loc[6]['cutnum']) * 256
fid.seek(pt_radial_block, 0)
a = 0
for i in range(1, 12):
	f['radinfo'].loc[i-1] = {}
	f['radinfo'].loc[i-1][str(i)] = {}
	# ------radial state
	f['radinfo'].loc[i-1][str(i)]['state'] = []
	# ------spot blank
	f['radinfo'].loc[i-1][str(i)]['spotblank'] = []
	# ------sequence number
	f['radinfo'].loc[i-1][str(i)]['seqnum'] = []
	# ------radial number
	f['radinfo'].loc[i-1][str(i)]['curnum'] = []
	# ------elevation number
	f['radinfo'].loc[i-1][str(i)]['elenum'] = []
	# ------azimuth
	f['radinfo'].loc[i-1][str(i)]['azi'] = []
	# ------elevation
	f['radinfo'].loc[i-1][str(i)]['ele'] = []
	# ------seconds
	f['radinfo'].loc[i-1][str(i)]['sec'] = []
	# ------microseconds
	f['radinfo'].loc[i-1][str(i)]['micro'] = []
	# ------length of data
	f['radinfo'].loc[i-1][str(i)]['datalen'] = []
	# ------moment number
	f['radinfo'].loc[i-1][str(i)]['momnum'] = []
	f['radinfo'].loc[i-1][str(i)]['reserved'] = []

#print(f['radinfo'].loc[0]['1']['state'])
for i in range(1, 12):
	f['dbt'].loc[i] = {}
	f['dbz'].loc[i] = {}
	f['zdr'].loc[i] = {}
	f['cc'].loc[i] = {}
	f['phidp'].loc[i] = {}
	f['kdp'].loc[i] = {}

while True:
	rad_dict = {}
	# ------radial state
	state = np.fromfile(fid, np.int32, 1)[0]
	# ------spot blank
	spotblank = np.fromfile(fid, np.int32, 1)[0]
	# ------sequence number
	seqnum = np.fromfile(fid, np.int32, 1)[0]
	# ------radial number
	curnum = np.fromfile(fid, np.int32, 1)[0]
	# ------elevation number
	elenum = np.fromfile(fid, np.int32, 1)[0]

	f['radinfo'].loc[int(elenum - 1)][str(int(elenum))]['state'].append(state)
	f['radinfo'].loc[int(elenum - 1)][str(int(elenum))]['spotblank'].append(spotblank)
	f['radinfo'].loc[int(elenum - 1)][str(int(elenum))]['seqnum'].append(seqnum)
	f['radinfo'].loc[int(elenum - 1)][str(int(elenum))]['curnum'].append(curnum)
	f['radinfo'].loc[int(elenum - 1)][str(int(elenum))]['elenum'].append(elenum)

	# ------azimuth
	f['radinfo'].loc[int(elenum - 1)][str(int(elenum))]['azi'].append(np.fromfile(fid, np.float32, 1)[0])
	# ------elevation
	f['radinfo'].loc[int(elenum - 1)][str(int(elenum))]['ele'].append(np.fromfile(fid, np.float32, 1)[0])
	# ------seconds
	f['radinfo'].loc[int(elenum - 1)][str(int(elenum))]['sec'].append(np.fromfile(fid, np.int32, 1)[0])
	# ------microseconds
	f['radinfo'].loc[int(elenum - 1)][str(int(elenum))]['micro'].append(np.fromfile(fid, np.int32, 1)[0])
	# ------length of data
	datalen = np.fromfile(fid, np.int32, 1)[0]
	f['radinfo'].loc[int(elenum - 1)][str(int(elenum))]['datalen'].append(datalen)
	# ------moment number
	num_moment = np.fromfile(fid, np.int32, 1)[0]
	#print(num_moment)
	f['radinfo'].loc[int(elenum - 1)][str(int(elenum))]['momnum'].append(num_moment)
	f['radinfo'].loc[int(elenum - 1)][str(int(elenum))]['reserved'].append(np.fromfile(fid, np.int8, 20))

	radcnt = curnum
	b = 0
	for n in range(0, num_moment):
		# ------data type
		var_type = np.fromfile(fid, np.int32, 1)[0]
		# ------scale
		scale = np.fromfile(fid, np.int32, 1)[0]
		# ------offset
		offset = np.fromfile(fid, np.int32, 1)[0]
		# ------!!! bin length: bytes for one bin storage
		binbytenum = np.fromfile(fid, np.int16, 1)[0]
		# ------flags
		flag = np.fromfile(fid, np.int16, 1)[0]
		# ------length
		bin_length = np.fromfile(fid, np.int32, 1)[0]
		reserved = np.fromfile(fid, np.int8, 12)

		bin_num = bin_length / binbytenum

		if binbytenum == 1:
			data_raw = np.fromfile(fid, np.uint8, int(bin_num))
		else:
			data_raw = np.fromfile(fid, np.uint16, int(bin_num))

		b += 1

		if elenum != 2 and elenum != 4:
			if var_type in variable_encode.keys():
				f[variable_encode[var_type]].loc[int(elenum)][str(int(radcnt))] = (data_raw - offset)/scale

	a += 1
	if state == 6 or state == 4:
		break


# 打印第一个仰角，第一个方位角上的kdp观测值
print([item for item in f['kdp'].loc[1]['1']])