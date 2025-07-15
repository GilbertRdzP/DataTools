import ROOT
import os
import numpy as np

#Uncomment the following line if using old WCSim versions
#ROOT.gSystem.Load(os.environ['WCSIMDIR'] + "/libWCSimRoot.so")

class WCSim:
    def __init__(self, tree):
        print("number of entries in the geometry tree: " + str(self.geotree.GetEntries()))
        self.geotree.GetEntry(0)
        self.geo = self.geotree.wcsimrootgeom
        self.num_pmts = self.geo.GetWCNumPMT()
        self.tree = tree
        self.nevent = self.tree.GetEntries()
        print("number of entries in the tree: " + str(self.nevent))
        # Get first event and trigger to prevent segfault when later deleting trigger to prevent memory leak
        self.tree.GetEvent(0)
        self.current_event = 0
        self.event = self.tree.wcsimrootevent
        self.ntrigger = self.event.GetNumberOfEvents()
        self.trigger = self.event.GetTrigger(0)
        self.current_trigger = 0

    def get_event(self, ev):
        # Delete previous triggers to prevent memory leak (only if file does not change)
        # triggers = [self.event.GetTrigger(i) for i in range(self.ntrigger)]
        # oldfile = self.tree.GetCurrentFile()
        self.event.ReInitialize() # since v1.12.20
        self.tree.GetEvent(ev)
        # if self.tree.GetCurrentFile() == oldfile:
        #     [t.Delete() for t in triggers]
        self.current_event = ev
        self.event = self.tree.wcsimrootevent
        self.ntrigger = self.event.GetNumberOfEvents()

    def get_trigger(self, trig):
        self.trigger = self.event.GetTrigger(trig)
        self.current_trigger = trig
        return self.trigger

    def get_first_trigger(self):
        first_trigger = 0
        first_trigger_time = 9999999.0
        for index in range(self.ntrigger):
            self.get_trigger(index)
            trigger_time = self.trigger.GetHeader().GetDate()
            if trigger_time < first_trigger_time:
                first_trigger_time = trigger_time
                first_trigger = index
        return self.get_trigger(first_trigger)

    def get_truth_info(self):  # deprecated: should now use get_event_info instead, leaving here for use with old files
        self.get_trigger(0)
        tracks = self.trigger.GetTracks()
        energy = []
        position = []
        direction = []
        pid = []
        for i in range(self.trigger.GetNtrack()):
            if tracks[i].GetParenttype() == 0 and tracks[i].GetFlag() == 0 and tracks[i].GetIpnu() in [22, 11, -11, 13,
                                                                                                       -13, 111]:
                pid.append(tracks[i].GetIpnu())
                position.append([tracks[i].GetStart(0), tracks[i].GetStart(1), tracks[i].GetStart(2)])
                direction.append([tracks[i].GetDir(0), tracks[i].GetDir(1), tracks[i].GetDir(2)])
                energy.append(tracks[i].GetE())
        return direction, energy, pid, position

    def get_event_info(self):
        self.get_trigger(0)
        tracks = self.trigger.GetTracks()
        # Primary particles with no parent are the initial simulation
        particles = [t for t in tracks if t.GetFlag() == 0 and t.GetParenttype() == 0]
        # Check there is exactly one particle with no parent:
        if len(particles) == 1:
            # Only one primary, this is the particle being simulated
            return {
                "pid": particles[0].GetIpnu(),
                "position": [particles[0].GetStart(i) for i in range(3)],
                "direction": [particles[0].GetDir(i) for i in range(3)],
                "energy": particles[0].GetE()
            }
        # Particle with flag -1 is the incoming neutrino or 'dummy neutrino' used for gamma
        # WCSim saves the gamma details (except position) in the neutrino track with flag -1
        neutrino = [t for t in tracks if t.GetFlag() == -1]
        # Check for dummy neutrino that actually stores a gamma that converts to e+ / e-
        isConversion = len(particles) == 2 and {p.GetIpnu() for p in particles} == {11, -11}
        if isConversion and len(neutrino) == 1 and neutrino[0].GetIpnu() == 22:
            return {
                "pid": 22,
                "position": [particles[0].GetStart(i) for i in range(3)], # e+ / e- should have same position
                "direction": [neutrino[0].GetDir(i) for i in range(3)],
                "energy": neutrino[0].GetE()
            }
        # Check for dummy neutrino from old gamma simulations that didn't save the gamma info
        if isConversion and len(neutrino) == 1 and neutrino[0].GetIpnu() == 12 and neutrino[0].GetE() < 0.0001:
            # Should be a positron/electron pair from a gamma simulation (temporary hack since no gamma truth saved)
            momentum = [sum(p.GetDir(i) * p.GetP() for p in particles) for i in range(3)]
            norm = np.sqrt(sum(p ** 2 for p in momentum))
            return {
                "pid": 22,
                "position": [particles[0].GetStart(i) for i in range(3)],  # e+ / e- should have same position
                "direction": [p / norm for p in momentum],
                "energy": sum(p.GetE() for p in particles)
            }

        # Workaround for opticalphoton sim
        opticalphotons = [t for t in tracks if t.GetFlag() == -2]  # Flag==-1 doesn't have proper position
        if len(opticalphotons) == 1 and opticalphotons[0].GetIpnu() == 0:
            position = [opticalphotons[0].GetStop(i) for i in range(3)] 
            
        opticalphotons = [t for t in tracks if t.GetFlag() == -1]     
        if len(opticalphotons) == 1 and opticalphotons[0].GetIpnu() == 0:
            direction = [opticalphotons[0].GetDir(i) for i in range(3)]
            energy = opticalphotons[0].GetE()
            
            return {
                "pid": -22,
                "position": position,
                "direction": direction,
                "energy": energy
            }
            
        # Otherwise something else is going on... guess info from the primaries
        momentum = [sum(p.GetDir(i) * p.GetP() for p in particles) for i in range(3)]
        norm = np.sqrt(sum(p ** 2 for p in momentum))
        return {
            "pid": 0,  # there's more than one particle so just use pid 0
            "position": [sum(p.GetStart(i) for p in particles)/len(particles) for i in range(3)],  # average position
            "direction": [p / norm for p in momentum],  # direction of sum of momenta
            "energy": sum(p.GetE() for p in particles)  # sum of energies
        }

    def get_digitized_hits(self):
        position = []
        charge = []
        time = []
        pmt = []
        trigger = []
        for t in range(self.ntrigger):
            self.get_trigger(t)
            trig_type = self.trigger.GetTriggerType()
            triggerInfo = self.trigger.GetTriggerInfo() 
            if (trig_type != 3) and (len(triggerInfo) >= 3):
                triggerShift = triggerInfo[1]
                triggerTime = triggerInfo[2]

            for hit in self.trigger.GetCherenkovDigiHits():
                pmt_id = hit.GetTubeId() - 1
                position.append([self.geo.GetPMT(pmt_id).GetPosition(j) for j in range(3)])
                charge.append(hit.GetQ())
                time.append(hit.GetT() + triggerTime - triggerShift) ## Correct the hit time adding the trigger time and resting the trigger shift.
                pmt.append(pmt_id)
                trigger.append(t)
        hits = {
            "position": np.asarray(position, dtype=np.float32),
            "charge": np.asarray(charge, dtype=np.float32),
            "time": np.asarray(time, dtype=np.float32),
            "pmt": np.asarray(pmt, dtype=np.int32),
            "trigger": np.asarray(trigger, dtype=np.int32)
        }
        return hits

    def get_true_hits(self):
        position = []
        track = []
        pmt = []
        PE = []
        trigger = []
        for t in range(self.ntrigger):
            self.get_trigger(t)
            for hit in self.trigger.GetCherenkovHits():
                pmt_id = hit.GetTubeID() - 1
                tracks = set()
                for j in range(hit.GetTotalPe(0), hit.GetTotalPe(0)+hit.GetTotalPe(1)):
                    pe = self.trigger.GetCherenkovHitTimes().At(j)
                    tracks.add(pe.GetParentID())
                position.append([self.geo.GetPMT(pmt_id).GetPosition(k) for k in range(3)])
                track.append(tracks.pop() if len(tracks) == 1 else -2)
                pmt.append(pmt_id)
                PE.append(hit.GetTotalPe(1))
                trigger.append(t)
        hits = {
            "position": np.asarray(position, dtype=np.float32),
            "track": np.asarray(track, dtype=np.int32),
            "pmt": np.asarray(pmt, dtype=np.int32),
            "PE": np.asarray(PE, dtype=np.int32),
            "trigger": np.asarray(trigger, dtype=np.int32)
        }
        return hits

    def get_hit_photons(self):
        start_position = []
        end_position = []
        start_dir = []
        end_dir = []
        start_time = []
        end_time = []
        track = []
        pmt = []
        trigger = []
        for t in range(self.ntrigger):
            self.get_trigger(t)
            n_photons = self.trigger.GetNcherenkovhittimes()
            trigger.append(np.full(n_photons, t, dtype=np.int32))
            counts = [h.GetTotalPe(1) for h in self.trigger.GetCherenkovHits()]
            hit_pmts = [h.GetTubeID()-1 for h in self.trigger.GetCherenkovHits()]
            pmt.append(np.repeat(hit_pmts, counts))
            end_time.append(np.zeros(n_photons, dtype=np.float32))
            track.append(np.zeros(n_photons, dtype=np.int32))
            start_time.append(np.zeros(n_photons, dtype=np.float32))
            start_position.append(np.zeros((n_photons, 3), dtype=np.float32))
            end_position.append(np.zeros((n_photons, 3), dtype=np.float32))
            start_dir.append(np.zeros((n_photons, 3), dtype=np.float32))
            end_dir.append(np.zeros((n_photons, 3), dtype=np.float32))
            photons = self.trigger.GetCherenkovHitTimes()
            for it, p in enumerate(photons):
                end_time[t][it] = p.GetTruetime()
                track[t][it] = p.GetParentID()
                try:  # Only works with new tracking branch of WCSim
                    start_time[t][it] = p.GetPhotonStartTime()
                    for i in range(3):
                        start_position[t][it,i] = p.GetPhotonStartPos(i)/10
                        end_position[t][it,i] = p.GetPhotonEndPos(i)/10
                        start_dir[t][it,i] = p.GetPhotonStartDir(i)
                        end_dir[t][it,i] = p.GetPhotonEndDir(i)
                except AttributeError: # leave as zeros if not using tracking branch
                    pass
        photons = {
            "start_position": np.concatenate(start_position),
            "end_position": np.concatenate(end_position),
            "start_dir": np.concatenate(start_dir),
            "end_dir": np.concatenate(end_dir),
            "start_time": np.concatenate(start_time),
            "end_time": np.concatenate(end_time),
            "track": np.concatenate(track),
            "pmt": np.concatenate(pmt),
            "trigger": np.concatenate(trigger)
        }
        return photons

    def get_tracks(self):
        track_id = []
        pid = []
        start_time = []
        energy = []
        start_position = []
        stop_position = []
        dir = []
        parent = []
        flag = []
#        boundary_points = []
        boundary_kes = []
        boundary_times = []
        boundary_types = []
        for t in range(self.ntrigger):
            self.get_trigger(t)
            for track in self.trigger.GetTracks():
                track_id.append(track.GetId())
                pid.append(track.GetIpnu())
                start_time.append(track.GetTime())
                energy.append(track.GetE())
                start_position.append([track.GetStart(i) for i in range(3)])
                stop_position.append([track.GetStop(i) for i in range(3)])
                dir.append([track.GetDir(i) for i in range(3)])
                parent.append(track.GetParenttype())
                flag.append(track.GetFlag())
#                boundary_points.append(np.array([b for b in track.GetBoundaryPoints()],dtype=np.float32))
                boundary_times.append(np.array([b for b in track.GetBoundaryTimes()],dtype=np.float32))
                boundary_kes.append(np.array([b for b in track.GetBoundaryKEs()],dtype=np.float32))
                boundary_types.append(np.asarray([b for b in track.GetBoundaryTypes()], dtype=np.int32))
        tracks = {
            "id": np.asarray(track_id, dtype=np.int32),
            "pid": np.asarray(pid, dtype=np.int32),
            "start_time": np.asarray(start_time, dtype=np.float32),
            "energy": np.asarray(energy, dtype=np.float32),
            "start_position": np.asarray(start_position, dtype=np.float32),
            "stop_position": np.asarray(stop_position, dtype=np.float32),
            "dir": np.asarray(dir, dtype=np.float32),
            "parent": np.asarray(parent, dtype=np.int32),
            "flag": np.asarray(flag, dtype=np.int32),
#            "boundary_points": np.asarray(boundary_points, dtype=object),
            "boundary_times": np.asarray(boundary_times, dtype=object),
            "boundary_kes": np.asarray(boundary_kes, dtype=object),
            "boundary_types": np.array(boundary_types, dtype=object),
        }
        return tracks

    def get_triggers(self):
        trigger_times = np.empty(self.ntrigger, dtype=np.float32)
        trigger_types = np.empty(self.ntrigger, dtype=np.int32)
        for t in range(self.ntrigger):
            self.get_trigger(t)
            trigger_times[t] = self.trigger.GetHeader().GetDate()
            trig_type = self.trigger.GetTriggerType()
            if trig_type > np.iinfo(np.int32).max:
                trig_type = -1;
            trigger_types[t] = trig_type
        triggers = {
                "time": trigger_times,
                "type": trigger_types
        }
        return triggers


class WCSimFile(WCSim):
    def __init__(self, filename):
        self.file = ROOT.TFile(filename, "read")
        tree = self.file.Get("wcsimT")
        self.geotree = self.file.Get("wcsimGeoT")
        super().__init__(tree)

    def __del__(self):
        self.file.Close()


class WCSimChain(WCSim):
    def __init__(self, filenames):
        self.chain = ROOT.TChain("wcsimT")
        for file in filenames:
            self.chain.Add(file)
        self.file = self.GetFile()
        self.geotree = self.file.Get("wcsimGeoT")
        super().__init__(self.chain)

def get_label(infile):
    if "_gamma" in infile:
        label = 0
    elif "_e" in infile:
        label = 1
    elif "_mu" in infile:
        label = 2
    elif "_pi0" in infile:
        label = 3
    else:
        print("Unknown input file particle type")
        raise SystemExit
    return label
