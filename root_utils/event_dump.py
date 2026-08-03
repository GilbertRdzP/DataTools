"""
Python 3 script for processing a list of ROOT files into .npz files

To keep references to the original ROOT files, the file path is stored in the output.
An index is saved for every event in the output npz file corresponding to the event index within that ROOT file (ev).

Authors: Nick Prouse
"""

import argparse
from root_utils.root_file_utils import *
from root_utils.pos_utils import *

ROOT.gROOT.SetBatch(True)


def get_args():
    parser = argparse.ArgumentParser(description='dump WCSim data into numpy .npz file')
    parser.add_argument('input_files', type=str, nargs='+')
    parser.add_argument('-d', '--output_dir', type=str, default=None)
    args = parser.parse_args()
    return args


def dump_file(infile, outfile):

    wcsim = WCSimFile(infile)
    nevents = wcsim.nevent - 1 #Remove the last event which is corrupt (Use only for beam)

    # All data arrays are initialized here

    event_id = np.empty(nevents, dtype=np.int32)
    root_file = np.empty(nevents, dtype=object)

    pid = np.empty(nevents, dtype=np.int32)
    position = np.empty((nevents, 3), dtype=np.float64)
    direction = np.empty((nevents, 3), dtype=np.float64)
    energy = np.empty(nevents,dtype=np.float64)

    digi_hit_pmt = np.empty(nevents, dtype=object)
    digi_hit_charge = np.empty(nevents, dtype=object)
    digi_hit_time = np.empty(nevents, dtype=object)
    digi_hit_trigger = np.empty(nevents, dtype=object)

    true_hit_pmt = np.empty(nevents, dtype=object)
    true_hit_time = np.empty(nevents, dtype=object)
    true_hit_pos = np.empty(nevents, dtype=object)
    true_hit_dir = np.empty(nevents, dtype=object)
    true_hit_start_time = np.empty(nevents, dtype=object)
    true_hit_start_pos = np.empty(nevents, dtype=object)
    true_hit_start_dir = np.empty(nevents, dtype=object)
    true_hit_parent = np.empty(nevents, dtype=object)

    track_id = np.empty(nevents, dtype=object)
    track_pid = np.empty(nevents, dtype=object)
    track_start_time = np.empty(nevents, dtype=object)
    track_energy = np.empty(nevents, dtype=object)
    track_start_position = np.empty(nevents, dtype=object)
    track_stop_position = np.empty(nevents, dtype=object)
    track_dir = np.empty(nevents, dtype=object)
    track_parent = np.empty(nevents, dtype=object)
    track_flag = np.empty(nevents, dtype=object)
#    track_boundary_points = np.empty(nevents, dtype=object)
    track_boundary_times = np.empty(nevents, dtype=object)
    track_boundary_kes = np.empty(nevents, dtype=object)
    track_boundary_types = np.empty(nevents, dtype=object)

    trigger_time = np.empty(nevents, dtype=object)
    trigger_type = np.empty(nevents, dtype=object)

    # Optional RooTracker information.
    # These arrays are only saved if at least one event returns a RooTracker dict.
    evt_code = np.empty(nevents, dtype=np.int32)
    neutrino_id = np.empty(nevents, dtype=np.int32)
    npions = np.empty(nevents, dtype=np.int32)

    evt_code[:] = -9999
    neutrino_id[:] = 0
    npions[:] = -1

    has_roostracker_info = False

    # Optional outer detector information.
    # A real event with no OD hits is stored as 0; -1 means unavailable.
    od_nhits = np.full(nevents, -1, dtype=np.int32)
    has_od_info = False

    for ev in range(wcsim.nevent - 1): #Remove the last event which is corrupt (Use only for beam)
        wcsim.get_event(ev)

        event_info = wcsim.get_event_info()
        pid[ev] = event_info["pid"]
        position[ev] = event_info["position"]
        direction[ev] = event_info["direction"]
        energy[ev] = event_info["energy"]

        # Optional RooTracker information.
        roostracker_info = wcsim.get_roostracker_event_info()
        if roostracker_info is not None:
            has_roostracker_info = True
            evt_code[ev] = roostracker_info["evt_code"]
            neutrino_id[ev] = roostracker_info["neutrino_id"]
            npions[ev] = roostracker_info["npions"]

        event_od_nhits = wcsim.get_od_nhits()
        if event_od_nhits is not None:
            has_od_info = True
            od_nhits[ev] = event_od_nhits

        true_hits = wcsim.get_hit_photons()
        true_hit_pmt[ev] = true_hits["pmt"]
        true_hit_time[ev] = true_hits["end_time"]
        true_hit_pos[ev] = true_hits["end_position"]
        true_hit_dir[ev] = true_hits["end_dir"]
        true_hit_start_time[ev] = true_hits["start_time"]
        true_hit_start_pos[ev] = true_hits["start_position"]
        true_hit_start_dir[ev] = true_hits["start_dir"]
        true_hit_parent[ev] = true_hits["track"]

        digi_hits = wcsim.get_digitized_hits()
        digi_hit_pmt[ev] = digi_hits["pmt"]
        digi_hit_charge[ev] = digi_hits["charge"]
        digi_hit_time[ev] = digi_hits["time"]
        digi_hit_trigger[ev] = digi_hits["trigger"]

        tracks = wcsim.get_tracks()
        track_id[ev] = tracks["id"]
        track_pid[ev] = tracks["pid"]
        track_start_time[ev] = tracks["start_time"]
        track_energy[ev] = tracks["energy"]
        track_start_position[ev] = tracks["start_position"]
        track_stop_position[ev] = tracks["stop_position"]
        track_dir[ev] = tracks["dir"]
        track_parent[ev] = tracks["parent"]
        track_flag[ev] = tracks["flag"]
#        track_boundary_points[ev] = tracks["boundary_points"]
        track_boundary_times[ev] = tracks["boundary_times"]
        track_boundary_kes[ev] = tracks["boundary_kes"]
        track_boundary_types[ev] = tracks["boundary_types"]

        triggers = wcsim.get_triggers()
        trigger_time[ev] = triggers["time"]
        trigger_type[ev] = triggers["type"]

        event_id[ev] = ev
        root_file[ev] = infile

    output_dict = dict(
        event_id=event_id,
        root_file=root_file,

        pid=pid,
        position=position,
        direction=direction,
        energy=energy,

        digi_hit_pmt=digi_hit_pmt,
        digi_hit_charge=digi_hit_charge,
        digi_hit_time=digi_hit_time,
        digi_hit_trigger=digi_hit_trigger,

        true_hit_pmt=true_hit_pmt,
        true_hit_time=true_hit_time,
        true_hit_pos=true_hit_pos,
        true_hit_dir=true_hit_dir,
        true_hit_start_time=true_hit_start_time,
        true_hit_start_pos=true_hit_start_pos,
        true_hit_start_dir=true_hit_start_dir,
        true_hit_parent=true_hit_parent,

        track_id=track_id,
        track_pid=track_pid,
        track_start_time=track_start_time,
        track_energy=track_energy,
        track_start_position=track_start_position,
        track_stop_position=track_stop_position,
        track_dir=track_dir,
        track_parent=track_parent,
        track_flag=track_flag,
        # track_boundary_points=track_boundary_points,
        track_boundary_times=track_boundary_times,
        track_boundary_kes=track_boundary_kes,
        track_boundary_types=track_boundary_types,

        trigger_time=trigger_time,
        trigger_type=trigger_type,
    )

    if has_roostracker_info:
        output_dict["evt_code"] = evt_code
        output_dict["neutrino_id"] = neutrino_id
        output_dict["npions"] = npions

    if has_od_info:
        output_dict["od_nhits"] = od_nhits

    np.savez_compressed(outfile, **output_dict)
    
    del wcsim


if __name__ == '__main__':

    config = get_args()
    if config.output_dir is not None:
        print("output directory: " + str(config.output_dir))
        if not os.path.exists(config.output_dir):
            print("                  (does not exist... creating new directory)")
            os.mkdir(config.output_dir)
        if not os.path.isdir(config.output_dir):
            raise argparse.ArgumentTypeError("Cannot access or create output directory" + config.output_dir)
    else:
        print("output directory not provided... output files will be in same locations as input files")

    file_count = len(config.input_files)
    current_file = 0

    for input_file in config.input_files:
        if os.path.splitext(input_file)[1].lower() != '.root':
            print("File " + input_file + " is not a .root file, skipping")
            continue
        input_file = os.path.abspath(input_file)

        if config.output_dir is None:
            output_file = os.path.splitext(input_file)[0] + '.npz'
        else:
            output_file = os.path.join(config.output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.npz')

        print("\nNow processing " + input_file)
        print("Outputting to " + output_file)

        dump_file(input_file, output_file)

        current_file += 1
        print("Finished converting file " + output_file + " (" + str(current_file) + "/" + str(file_count) + ")")

    print("\n=========== ALL FILES CONVERTED ===========\n")
