import ROOT
import os
import numpy as np

#Uncomment the following line if using old WCSim versions
#ROOT.gSystem.Load(os.environ['WCSIMDIR'] + "/libWCSimRoot.so")
ROOT.gSystem.Load("/project/6002456/gilrdzp/Analysis/RecoAnalysis/libReadNuPRISM/libReadNuPRISM.so")

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

    def _truth_from_roostracker(self):
        """
        Build truth information from fRooTrackerOutputTree.
    
        Truth definitions:
          energy    = selected particle energy in MeV
          position  = EvtVtx[0:3] in cm
          direction = selected particle momentum direction
    
        Event PID rules:
          IsTrueNCGamma                   -> 22
          IsTrueNuECC0PiEvent             -> 11
          IsUncontainedTrueNuMuCC0PiEvent -> 13
          IsTrueNCPi0                     -> 111
          else                            -> 0
        """
        if not hasattr(self, "file") or not self.file:
            return None
    
        rt = self.file.Get("fRooTrackerOutputTree")
        if not rt:
            return None
    
        if self.current_event >= rt.GetEntries():
            return None
    
        rt.GetEntry(self.current_event)
    
        try:
            vtx = rt.NRooTrackerVtx
        except AttributeError:
            return None
    
        if not vtx:
            return None
    
        # ------------------------------------------------------------
        # Basic helpers
        # ------------------------------------------------------------
    
        def get_neut_code(vtx):
            """
            Extract NEUT interaction code from RooTracker EvtCode.
            """
            try:
                evt_code = int(vtx.EvtCode.GetString().Atoi())
            except Exception:
                try:
                    evt_code = int(vtx.EvtCode.GetString().Data())
                except Exception:
                    evt_code = None
            return evt_code
    
        def p4_gev(i):
            """
            Access StdHepP4 as a flat array:
              StdHepP4[4*i + 0] = px
              StdHepP4[4*i + 1] = py
              StdHepP4[4*i + 2] = pz
              StdHepP4[4*i + 3] = E
            Units are assumed to be GeV.
            """
            px = float(vtx.StdHepP4[4*i + 0])
            py = float(vtx.StdHepP4[4*i + 1])
            pz = float(vtx.StdHepP4[4*i + 2])
            E  = float(vtx.StdHepP4[4*i + 3])
            return px, py, pz, E
    
        def momentum_mag_gev(i):
            px, py, pz, _ = p4_gev(i)
            return np.sqrt(px*px + py*py + pz*pz)
    
        def ring_evis_gev(mom, mass):
            """
            Same visible-energy estimate as the C++ code.
            """
            n_water = 1.334
            threshold = mass / np.sqrt(n_water*n_water - 1.0)
    
            e = np.sqrt(mom*mom + mass*mass)
            e_thr = np.sqrt(threshold*threshold + mass*mass)
    
            return max(0.0, e - e_thr)
    
        def mass_for_one_ring_candidate(iPDG, mom):
            """
            Python equivalent of the mass assignment inside IsOneRingCandidate.
            Returns None if the particle should be skipped.
            """
            if iPDG == 22 and mom > 0.01: #gamma excl. low mom
                return 0.0
            elif iPDG == 11:              #electron
                return 0.0005110
            elif iPDG == 13:              #muon
                return 0.1056584
            elif iPDG == 211:             #pi+-
                return 0.1395702
            elif iPDG == 111:             #pi0
                return 0.0
            elif iPDG == 2212:            #proton
                return 0.9382720
            elif iPDG == 221:             #eta, decays to pi0             
                return 0.0
            elif iPDG in (321, 310, 130): #Kaons - decay to muons/pions with short lifetime, set to 0 threshold for now...
                return 0.0
            else:
                return None
    
        def is_one_ring_candidate(thresh):
            """
            Python equivalent of:
    
              bool IsOneRingCandidate(int &pid, double thresh)
    
            Returns:
              (is_one_ring, pid, index)
    
            where pid is the absolute PDG of the one visible particle,
            and index is its StdHep index.
            """
            n_visible = 0
            candidate_pid = 0
            candidate_index = None
    
            for i in range(3, int(vtx.StdHepN)):
                iPDG = abs(int(vtx.StdHepPdg[i]))
    
                mom = momentum_mag_gev(i)
                mass = mass_for_one_ring_candidate(iPDG, mom)
    
                if mass is None:
                    continue
    
                ring_evis = ring_evis_gev(mom, mass)
    
                if ring_evis > thresh:
                    n_visible += 1
                    candidate_pid = iPDG
                    candidate_index = i
    
            return (n_visible == 1), candidate_pid, candidate_index
    
        def is_true_numu_cc0pi_event(thresh):
            """
            Python equivalent of IsTrueNuMuCC0PiEvent.
            """
            neut_code = get_neut_code(vtx)
            if neut_code is None:
                return False
    
            if abs(neut_code) >= 30 or int(vtx.StdHepPdg[0]) != 14:
                return False
            one_ring, _, _ = is_one_ring_candidate(thresh)
            if not one_ring:
                return False
    
            return True
    
        def is_true_nue_cc0pi_event(thresh):
            """
            Python equivalent of IsTrueNuECC0PiEvent.
            """
            neut_code = get_neut_code(vtx)
            if neut_code is None:
                return False
            if abs(neut_code) >= 30 or int(vtx.StdHepPdg[0]) != 12:
                return False
            one_ring, _, _ = is_one_ring_candidate(thresh)
            if one_ring:
                return True
    
            return False
    
        def is_true_ncpi0():
            """
            Python equivalent of IsTrueNCPi0.
            Uses the same fixed 30 MeV visible-energy threshold.
            """
            neut_code = get_neut_code(vtx)
            if neut_code is None:
                return False
    
            if abs(neut_code) < 30:
                return False
    
            n_pi0 = 0
            for i in range(0, int(vtx.StdHepN)):
                if int(vtx.StdHepPdg[i]) == 111:
                    n_pi0 += 1
    
            n_charge = 0
            n_photon = 0
    
            if n_pi0:
                for i in range(0, int(vtx.StdHepN)):
                    iPDG = abs(int(vtx.StdHepPdg[i]))
                    mom = momentum_mag_gev(i)
    
                    if iPDG == 22:
                        mass = 0.0
                    elif iPDG == 11:
                        mass = 0.0005110
                    elif iPDG == 13:
                        mass = 0.1056584
                    elif iPDG == 111:
                        mass = 0.0
                    elif iPDG == 211:
                        mass = 0.1395702
                    elif iPDG == 2212:
                        mass = 0.9382720
                    else:
                        mass = 0.9382720
    
                    ring_evis = ring_evis_gev(mom, mass)
    
                    if ring_evis > 0.03:
                        if iPDG == 22 or iPDG == 111:
                            n_photon += 1
                        else:
                            n_charge += 1
    
            return (n_pi0 >= 1) and (n_charge == 0)
    
        def is_true_ncgamma(thresh):
            """
            Python implementation of a IsTrueNCGamma method.
            """
            neut_code = get_neut_code(vtx)
            if neut_code is None:
                return False
    
            if abs(neut_code) < 30:
                return False
    
            # Explicit pion veto
            for i in range(3, int(vtx.StdHepN)):
                iPDG = abs(int(vtx.StdHepPdg[i]))
                if iPDG == 111 or iPDG == 211:
                    return False
    
            one_ring, ring_pid, _ = is_one_ring_candidate(thresh)
            if not one_ring:
                return False
    
            if ring_pid != 22:
                return False
    
            return True
    
        # ------------------------------------------------------------
        # Event PID classification
        # ------------------------------------------------------------
        thresh = 0.03  # GeV = 30 MeV
    
        if is_true_ncgamma(thresh):
            event_pid = 22
        elif is_true_nue_cc0pi_event(thresh):
            event_pid = 11
        elif is_true_numu_cc0pi_event(thresh):
            event_pid = 13
        elif is_true_ncpi0():
            event_pid = 111
        else:
            event_pid = 0
    
        # ------------------------------------------------------------
        # Position
        # ------------------------------------------------------------
        true_position = [
            float(vtx.EvtVtx[0]) * 100.0,
            float(vtx.EvtVtx[1]) * 100.0,
            float(vtx.EvtVtx[2]) * 100.0,
        ]
    
        # ------------------------------------------------------------
        # Direction and energy selection
        #
        # Keep your current logic:
        #   1. Prefer charged lepton if present
        #   2. Otherwise use highest-energy pion/gamma candidate
        # ------------------------------------------------------------
    
        particle_p3 = None
        true_energy = None
        pion_gamma_candidates = []
    
        for i in range(1, int(vtx.StdHepN)):
            pdg = int(vtx.StdHepPdg[i])
    
            px = float(vtx.StdHepP4[4*i + 0]) * 1000.0
            py = float(vtx.StdHepP4[4*i + 1]) * 1000.0
            pz = float(vtx.StdHepP4[4*i + 2]) * 1000.0
            E  = float(vtx.StdHepP4[4*i + 3]) * 1000.0
    
            if abs(pdg) in (11, 13, 15):
                true_energy = E
                particle_p3 = [px, py, pz]
                break
    
            elif abs(pdg) in (211, 111, 213, 113, 22):
                pion_gamma_candidates.append({
                    "index": i,
                    "pdg": pdg,
                    "p4": [px, py, pz, E],
                    "energy": E,
                })
    
        # If no charged lepton was found, use the highest energy pion/gamma
        if particle_p3 is None:
            if len(pion_gamma_candidates) == 0:
                return None
    
            best_particle = max(
                pion_gamma_candidates,
                key=lambda x: x["energy"]
            )
    
            px, py, pz, E = best_particle["p4"]
            true_energy = E
            particle_p3 = [px, py, pz]
    
        if particle_p3 is None:
            return None
    
        norm = np.sqrt(sum(p ** 2 for p in particle_p3))
    
        if norm == 0 or not np.isfinite(norm):
            return None
    
        true_direction = [p / norm for p in particle_p3]
    
        return {
            "pid": event_pid,
            "position": true_position,
            "direction": true_direction,
            "energy": true_energy,
        }

    def get_roostracker_event_info(self):
        """
        Return extra RooTracker truth information if fRooTrackerOutputTree exists.

        Returns
        -------
        dict or None
        Dictionary contains:
          - evt_code
          - neutrino_id
          - npions
        Returns None if the file has no RooTracker tree or the vertex cannot be read.
        """
        if not hasattr(self, "file") or not self.file:
            return None
        rt = self.file.Get("fRooTrackerOutputTree")
        if not rt:
            return None
        if self.current_event >= rt.GetEntries():
            return None
        rt.GetEntry(self.current_event)
        try:
            vtx = rt.NRooTrackerVtx
        except AttributeError:
            return None
        if not vtx:
            return None
            
        # NEUT event code:
        try:
            evt_code = int(vtx.EvtCode.GetString().Atoi())
        except Exception:
            try:
                evt_code = int(vtx.EvtCode.GetString().Data())
            except Exception:
                evt_code = None
        # Neutrino type ID:
        try:
            neutrino_id = int(vtx.StdHepPdg[0])
        except Exception:
            neutrino_id = None
        # Number of pions:
        npions = 0
        try:
            for i in range(1, int(vtx.StdHepN)):
                pdg_mod = abs(int(vtx.StdHepPdg[i])) % 1000

                if pdg_mod in (211, 111, 213, 113):
                    npions += 1
        except Exception:
            npions = None
        if evt_code is None:
            evt_code = -9999
        if neutrino_id is None:
            neutrino_id = 0
        if npions is None:
            npions = -1
        return {
            "evt_code": evt_code,
            "neutrino_id": neutrino_id,
            "npions": npions,
        }
    
    def get_event_info(self):
        self.get_trigger(0)
        tracks = self.trigger.GetTracks()

        roostracker_truth = self._truth_from_roostracker() ## If it contains a RooTracker class then do this 
        if roostracker_truth is not None:
            return roostracker_truth
        
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
            for hit in self.trigger.GetCherenkovDigiHits():
                pmt_id = hit.GetTubeId() - 1
                position.append([self.geo.GetPMT(pmt_id).GetPosition(j) for j in range(3)])
                charge.append(hit.GetQ())
                time.append(hit.GetT())
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
