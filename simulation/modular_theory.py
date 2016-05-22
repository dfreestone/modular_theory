#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Name   : modular_theory.py
Author : David Freestone (david.freestone@bucknell.edu)
Date   : 01.01.2016

This time_event module is covered under the modified BSD license
Copyright (c) 2015, David M. Freestone
All rights reserved.

Description:
"""

# What we need:
# [1] Main loop
#     [a]: update procedure state
#     [b]: update model state
#     [c]: update subject state

from pandas import DataFrame
from numba import jit
from numpy import array, zeros, arange, linspace
import numpy as np
import re

NUMBA_DISABLE_JIT = 1  # Disables JIT for debugging


class Simulation():

    def __init__(self, notation=None, resolution=0.01, parameters=None):

        self.parameters = {"beta_r": 0.5,
                           "beta_e": 1e-5,
                           "c": 1.0,
                           "alpha": 0.05,
                           "phi": 50.0,
                           "gamma": 0.2,
                           "r0": 0.05,
                           "A": 5.0,
                           "mu_poisson": 3.0,
                           "mu_wald": 0.68,
                           "lambda_wald": 0.93}
        if parameters is not None:
            self.parameters.update(parameters)

        # TODO(David): Completely change this, probably, because it sucks
        self.resolution = resolution
        self.original_notation = notation
        self.notation = notation  # in case we need to convert point events
        self.array_notation = notation  # numpy array for jit
        self.parse_notation()
        self.convert_notation()

        # TODO(David): Fix for multiple cycle types
        df_cycle_types = DataFrame(self.array_notation[:, :, 0], columns=["event", "id"])
        df_cycle_types = df_cycle_types[(df_cycle_types.event==STIMULUS_ON) | (df_cycle_types.event==STIMULUS_OFF)].drop_duplicates()
        df_cycle_types["row"] = arange(0, df_cycle_types.shape[0])

        self.time_marker_info = df_cycle_types.as_matrix()
        self.n_time_markers = self.time_marker_info.shape[0]
        self.time_marker_row = 2

    def parse_notation(self):
        """Return cycle information"""
        if self.notation is None:
            return None

        if not isinstance(self.notation, dict):
            self.notation = {'C1': self.notation}

        for cycle in self.notation:
            notation = self.convert_point_events(self.notation[cycle],
                                                 self.resolution)
            states = tuple([self.parse_state(state)
                            for state in notation.split()])
            self.notation[cycle] = {'notation': notation,
                                    'states': states}
        return None

    @staticmethod
    def convert_point_events(notation, resolution):
        """Return full notation, i.e., O.1 to O1 0.02T o1"""
        notation = notation.split()
        for i, state in enumerate(notation):
            if '.' in state:
                event, criteria = state.split('.')
                notation[i] = '{}{} {}T {}{}'.format(event, criteria,
                                                     resolution,
                                                     event.lower(), criteria)
        return ' '.join(notation)

    @staticmethod
    def parse_state(state_notation):
        """Split state inot event and criteria"""
        tokens = re.split(r'([0-9\.]+)', state_notation)
        state = [None, None]
        for token in tokens:
            if len(token) == 0:
                continue
            try:
                state[0] = float(token)
            except ValueError:
                state[1] = token
            #       TODO(David): Allow non-integer event IDs?
        if 'T' not in state and None not in state:
            state[0] = int(state[0])
        return tuple(state)

    def convert_notation(self):
        ncycles = len(self.notation)
        max_states = max([len(self.notation[cycle]['states'])
                          for cycle in self.notation])
        self.array_notation = zeros((max_states, 2, ncycles), dtype=float)
        for icy, cycle in enumerate(self.notation):
            for ist, state in enumerate(self.notation[cycle]['states']):
                criteria, event = state
                event = -ord(event[1]) if event == '~T' else ord(event)
                self.array_notation[ist, :, icy] = array([event, criteria])

    def run(self):
        (self.data, self.strength_memory,
         self.reference_memory, self.packet_initiation_rate, self.max_timestep) = __run__(self.array_notation, self.time_marker_info, self.n_time_markers, self.time_marker_row,
                                                                       self.parameters["beta_r"], self.parameters["beta_e"],
                                                                       self.parameters["c"], self.parameters["alpha"],
                                                                       self.parameters["phi"], self.parameters["gamma"],
                                                                       self.parameters["r0"], self.parameters["A"],
                                                                       self.parameters["mu_poisson"], self.parameters["mu_wald"],
                                                                       self.parameters["lambda_wald"])
        self.data = DataFrame(self.data, columns=["time", "event", "id"])
        self.data.event = self.data.event.astype(int).map(chr)

STIMULUS_ON = ord('S')
STIMULUS_OFF = ord('s')
RESPONSE_ON = ord('R')
RESPONSE_OFF = ord('r')
OUTCOME_ON = ord('O')
OUTCOME_OFF = ord('o')
TIME = ord('T')
RANDOM_TIME = -ord('T')
RANDOM_EXPONENTIAL = ord('E')

# indices for state matrix
event_idx = 0
criteria_idx = 1


@jit(nopython=True)
def __run__(CYCLE_TYPES, time_marker_info, n_time_markers, time_marker_row,
            beta_r, beta_e, c, alpha, phi, gamma,
            r0, A, mu_poisson, mu_wald, lambda_wald):
    # Experiment info
    time = 0
    resolution = 0.01
    session_duration = 120 * 60
    max_timesteps = 2 * int(session_duration / resolution)  # 2x for buffer

    data = zeros((int(max_timesteps), 3))
    data_count = 0

    response = True
    cycle = 0
    state = 0
    nstates = get_nstates(CYCLE_TYPES[:, :, 0])
    state_time = 0
    # TODO(David): This doesn't copy fully, so any change to current_cycle_info propogates... :-(
    cycle, nstates = get_next_cycle(CYCLE_TYPES)
    cycle_types = copy_matrix(CYCLE_TYPES)
    current_cycle_info = cycle_types[:, :, cycle]

    # model setup
    time_marker_started = zeros(int(n_time_markers))
    timestep = zeros(int(n_time_markers))
    max_timestep = zeros(int(n_time_markers))
    model_state = zeros(int(n_time_markers))

    # TODO(David): Optimize for C or R major ordering
    strength_memory = zeros((int(max_timesteps), int(n_time_markers)))
    scalar_strength_memory = zeros(int(n_time_markers))

    perceptual_store = linspace(0, 2*session_duration, int(max_timesteps))

    # TODO(David): Optimize for C or R major ordering
    reference_memory = zeros((int(max_timesteps), int(n_time_markers)))
    threshold = zeros(int(n_time_markers))

    reliability = zeros(int(n_time_markers))
    packet_initiation_rate = zeros(int(n_time_markers))
    total_packet_initiation_rate = zeros((int(max_timesteps), int(n_time_markers)))

    response_queue = zeros(int(max_timesteps))
    response_idx = 0
    response_count = 0
    max_response_count = int(max_timesteps)

    running = True
    while running:

        # update experiment
        time += resolution
        if time >= session_duration:
            running = False
            break

        # update procedure
        state_time += resolution
        outcome = False

        next_event = True
        while next_event:
            if state > nstates:
                state = 0
                state_time = 0
                cycle, nstates = get_next_cycle(CYCLE_TYPES)
                cycle_types = copy_matrix(CYCLE_TYPES)
                # TODO(David): This doesn't copy fully, so any change to current_cycle_info propogates... :-(
                current_cycle_info = cycle_types[:, :, cycle]

            event = int(current_cycle_info[state, event_idx])
            criteria = current_cycle_info[state, criteria_idx]

            if event == TIME:
                if state_time >= criteria:
                    state += 1
                    state_time = 0
                    data[data_count, 0] = time
                    data[data_count, 1] = TIME
                    data[data_count, 2] = criteria
                    data_count += 1
                else:
                    next_event = False

            elif event == RANDOM_TIME:
                # TODO(David): Implement me!
                current_cycle_info[state, event_idx] = RANDOM_EXPONENTIAL

            elif event == RESPONSE_ON:
                if response:
                    # TODO(David): Shoudl this record the end of this state?
                    #              So that we separate the response code from the state codes?
                    state += 1
                    state_time = 0
                else:
                    next_event = False

            elif event == RESPONSE_OFF:
                # TODO(David): Shoudl this record the end of this state?
                #              So that we separate the response code from the state codes?
                # TODO(David): Implement me!
                pass

            elif event == STIMULUS_ON:
                # NOTE(David): The [0] at the end is needed so that numba can represent the row as an int rather than a numpy array. Or something
                row = time_marker_info[(time_marker_info[:,0]==STIMULUS_ON) & (time_marker_info[:,1]==criteria), 2][0]
                time_marker_started[int(row)] = 1
                state += 1
                state_time = 0
                data[data_count, 0] = time
                data[data_count, 1] = STIMULUS_ON
                data[data_count, 2] = criteria
                data_count += 1

            elif event == STIMULUS_OFF:
                #NOTE(David): In reality, Modular Theory has no notion of stimulus state (on or off),
                #             It just has time markers. So there is no real "STIMULUS_OFF", there's just
                #             a time to reset a time marker (when it comes on)
                #             We'll keep this because its easier to work with both "S" and "s" this way
                row = time_marker_info[(time_marker_info[:,0]==STIMULUS_OFF) & (time_marker_info[:,1]==criteria), 2][0]
                time_marker_started[int(row)] = 1
                state += 1
                state_time = 0
                data[data_count, 0] = time
                data[data_count, 1] = STIMULUS_OFF
                data[data_count, 2] = criteria
                data_count += 1

            elif event == OUTCOME_ON:
                # TODO(David): Add this as a stimulus
                state += 1
                state_time = 0
                outcome = True
                data[data_count, 0] = time
                data[data_count, 1] = OUTCOME_ON
                data[data_count, 2] = criteria
                data_count += 1

            elif event == OUTCOME_OFF:
                # TODO(David): Add this as a stimulus
                state += 1
                state_time = 0
                outcome = False  # TODO(David): Make sure this actually works
                data[data_count, 0] = time
                data[data_count, 1] = OUTCOME_OFF
                data[data_count, 2] = criteria
                data_count += 1

            elif event == RANDOM_EXPONENTIAL:
                current_cycle_info[state, criteria_idx] = np.random.exponential(criteria)
                current_cycle_info[state, event_idx] = TIME
                data[data_count, 0] = time
                data[data_count, 1] = RANDOM_EXPONENTIAL
                data[data_count, 2] = criteria
                data_count += 1

        # update model state
        for time_marker in time_marker_info[:, 2]:

            """ update or reset time """
            if time_marker_started[int(time_marker)]:
                timestep[int(time_marker)] = 0
                time_marker_started[int(time_marker)] = 0
            else:
                timestep[int(time_marker)] += 1

            if outcome:
                """ Equation 1: clock """
                # TODO(David): we need to convert the timesteps to time, scale by c, then convert back to timestep
                #              but since model_state updates in real continous time, we don't want to update the
                #              timestep for real so it gets out of sync. So we'll create a temporary timestep
                perceived_time = c * (timestep[int(time_marker)] * resolution)
                temporary_timestep = int(perceived_time / resolution)

                """ Equation 2: perceptual store """
                # Using an already filled list that we just index into whenever we need to, e.g.

                """ Equation 3: reference memory """
                # TODO(David): Probably easiest thing to do is to make this a ufunc
                if temporary_timestep > max_timestep[int(time_marker)]:
                    reference_memory[max_timestep[int(time_marker)]:int(temporary_timestep), int(time_marker)] = perceptual_store[int(temporary_timestep):max_timestep[int(time_marker)]:-1]
                    max_timestep[int(time_marker)] = temporary_timestep
                for i in range(temporary_timestep+1):
                    reference_memory[i, int(time_marker)] = alpha*perceptual_store[int(temporary_timestep-i)] + (1-alpha)*reference_memory[i, int(time_marker)]

                """ Equation 4: threshold """
                # As long as there's reference_memory
                if max_timestep[int(time_marker)] > 0:
                    max_perceived_time = max_timestep[int(time_marker)] * resolution
                    random_threshold = phi + (phi*gamma)*np.random.randn()
                    # NOTE(David): numba now supports np.histogram 
                    histogram_reference_memory = histogram(reference_memory[:max_timestep[int(time_marker)], int(time_marker)], int(max_timestep[int(time_marker)]), max_perceived_time, resolution)
                    threshold[int(time_marker)] = percentile(histogram_reference_memory, resolution, random_threshold)

            """ Equation 5: state """
            model_state[int(time_marker)] = int(reference_memory[int(timestep[int(time_marker)]), int(time_marker)] < threshold[int(time_marker)])

            """ Equation 6: update strength memory """
            # NOTE(David): keeping a scalar_reference_memory means we don't need to store p_timestep anymore,
            #              and keeping a strength_memory means we can still store the entire time vector of strenghts
            beta = beta_r if outcome else beta_e
            scalar_strength_memory[int(time_marker)] += beta*(outcome - scalar_strength_memory[int(time_marker)]) * resolution
            strength_memory[int(timestep[int(time_marker)]), int(time_marker)] = scalar_strength_memory[int(time_marker)]

            """ Model update done  """

            """ Equation 7: operant rate """
            # Nothing to do here, its just a constant

            """ Equation 8: packet initation rate """
            packet_initiation_rate[int(time_marker)] = resolution*(r0 + (A * model_state[int(time_marker)] * strength_memory[int(timestep[int(time_marker)]), int(time_marker)]))
            total_packet_initiation_rate[int(timestep[int(time_marker)]), int(time_marker)] = packet_initiation_rate[int(time_marker)]

            # Update subject
            """New equation: summing packet initiation rates"""
            reliability[int(time_marker)] = reference_memory[0, int(time_marker)] if reference_memory[0, int(time_marker)] > 0 else 1

        # TODO(David): Note reliability is commented out below
        total_reliability = reliability.sum()
        combined_packet_initiation_rate = 0
        for i in range(n_time_markers):
            combined_packet_initiation_rate += packet_initiation_rate[i] * (reliability[i]/total_reliability)

        if np.random.rand() <= combined_packet_initiation_rate:
            """ Equation 9a: Poisson rv """
            P_u = np.random.poisson(mu_poisson)

            """ Equation 9b: Inverse Gaussian rv """
            # TODO(David): Make this a ufunc, prob.
            #              Numba does not have the function definition for adding
            #              the size argument
            if P_u > 0:
                rt = time  # TODO(David): Maybe clean this up too?
                for _ in range(P_u):
                    rt += np.random.wald(mu_wald, lambda_wald)
                    response_queue[response_count] = rt
                    response_count += 1
                    if response_count >= max_response_count:
                        # TODO(David): Repark the counters and junk
                        # For now, just erase all responses (bad!)
                        response_idx = 0
                        response_count = 0

        # TODO(David): Handle response_off in a reasonable way
        response = (response_count > 0) and time >= response_queue[response_idx]
        if response:
            # TODO(David): Add this as a stimulus
            response_idx += 1  # This procedure handles recording the response
            data[data_count, 0] = time
            data[data_count, 1] = RESPONSE_ON
            data[data_count, 2] = 1  # TODO(David): This does not account for more than 1 response
                                     #              Its 1 because if it were "criteria" like everything else,
                                     #              it would be wrong (it would be the criteria for whatever state we're in)
            data_count += 1
            if response_idx > response_count:
                response_idx = 0
                response_count = 0

    data = data[:data_count, :]
    return data, strength_memory, reference_memory, total_packet_initiation_rate, max_timestep


@jit(nopython=True)
def copy_matrix(src):
    dst = zeros(src.shape)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            for k in range(src.shape[2]):
                dst[i, j, k] = src[i, j, k]
    return dst


@jit(nopython=True)
def percentile(arr, resolution, p):
    n = arr.size
    np = (p/100)*n
    s = 0
    for i in range(n):
        s += arr[i]
        if s >= np:
            return (1+i) * resolution  # TODO(David): Make sure it actually is 1+i rather than i
    return n * resolution


# NOTE(David): Numba now supports np.histogram
@jit(nopython=True)
def histogram(arr_in, n, hmax, resolution):
    """ Modifies arr_out in place """
    nbins = int(hmax / resolution)
    # TODO(David): Should we allocate with zeros, or zero the array?
    arr_out = zeros(nbins)
    for i in range(n):
        bin_index = int(arr_in[i] / resolution)
        # TODO(David): Test if this if-else statement is event necessary
        if 0 <= bin_index < nbins:
            arr_out[int(bin_index)] += 1
        elif arr_in[i] >= hmax:
            arr_out[int(nbins)] += 1  # All other elements go into the last bin
    return arr_out


@jit(nopython=True)
def is_sorted(arr):
    for i in range(1, arr.size):
        if arr[i] < arr[i-1]:
            return False
    return True


@jit(nopython=True)
def quickSort(arr):
    """"  quickSort
      This function was modified translated from Darel Rex Finley's C implementation
    """
    MAX_LEVELS = 300

    beg = zeros(MAX_LEVELS)
    end = zeros(MAX_LEVELS)

    end[0] = arr.size

    i = 0
    while (i >= 0):
        L = int(beg[i])
        R = int(end[i]-1)

        if (L < R):
            piv = arr[L]

            while (L < R):
                while ((arr[R] >= piv) and (L < R)):
                    R -= 1
                if (L < R):
                    arr[L] = arr[R]
                    L += 1

                while ((arr[L] <= piv) and (L < R)):
                    L += 1
                if (L < R):
                    arr[R] = arr[L]
                    R -= 1

            arr[L] = piv
            beg[i+1] = L+1
            end[i+1] = end[i]
            end[i] = L
            i += 1
            if (end[i]-beg[i] > end[i-1]-beg[i-1]):
                swap = beg[i]
                beg[i] = beg[i-1]
                beg[i-1] = swap

                swap = end[i]
                end[i] = end[i-1]
                end[i-1] = swap
        else:
            i -= 1


@jit(nopython=True)
def __round__(x, dx):
    return round(x/dx) * dx


@jit(nopython=True)
def get_next_cycle(cycle_types):
    # TODO(David): Implement for real
    #               Need a way to put in the cycle as a cycle order
    cycle = 0
    nstates = get_nstates(cycle_types[:, :, cycle])
    return cycle, nstates


@jit(nopython=True)
def get_nstates(cycle_type):
    for state in range(cycle_type.shape[0]):
        if cycle_type[state, 1] == 0:
            return state
    return state

if __name__ == "__main__":
    procedure = "10T S1 60T O.1 s1"
    simulation = Simulation(procedure)
    simulation.run()
