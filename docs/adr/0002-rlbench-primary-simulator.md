# ADR 0002 — RLBench as primary simulator

Date: 2026-05-16

## Status

Accepted

## Context

The project needs a controlled simulation environment with single-arm manipulation tasks, RGB-D/segmentation/proprioception, expert demonstrations, and task customizability.

## Decision

Use RLBench as the primary simulator for P0/P1. Do not choose the simulator based on baseline convenience; choose it based on the ability to showcase programmatic constraint guidance and world-model rollouts.

## Consequences

RLBench/PyRep/CoppeliaSim setup becomes Milestone 0/1 work. Baselines should be adapted/reimplemented inside our RLBench setup instead of forcing the project into a baseline-native simulator.

## Alternatives considered

- ManiSkill/SAPIEN: potentially easier for CodeDiffuser-style baselines but less aligned with the chosen task/control story.
- DP3's existing environments: easier DP3 integration but less control over the desired constraint-composition story.
