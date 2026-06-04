# Data Processing

This document outlines the data processing pipeline for the social world model.

## Overview

This repository is designed to process pedestrian motion data from a wide range of datasets with various types of pedestrian data. The goal is to convert the data into a common format suitable for training and evaluating the social world model.

Minimal data required for a dataset to be included:

- Map of the environment (Occupancy grid)
- Position and orientation of each agent (person or robot) in the environment (x, y, theta) over time

Additional data that would be nice to have:

- 2D information extracted from images:
  - Bounding box or semantic segmentation of each visible agent
  - Facial expression of each agent.
    - Represented as a crop of face pixels
    - Processed by a facial expression recognition model to extract Facial Action Units (FAUs) or other emotion-related features.
  - Visual identifier of each agent. Useful for tracking across frames, even when the agent is not visible in some frames.
- 3D information extracted from LIDAR or depth sensors:
  - Bounding box of each agent (x, y, z, width, height, depth)
  - Body pose of each agent
- Semantic Information
  - Estimate of Current Gesture (Processed from a short history of body pose or raw image data)
  - Estimate of Group Membership (Processed from a short history of body pose or raw image data)
  - Estimate of Perception of Other Agents (Processed from a short history of body pose or raw image data)

(Potential) Source datasets are:

- CMU TBD Dataset
- Waymo Open Dataset
- UT CODA Dataset
- UT SCAND Dataset
- Nu Scenes Dataset
- Speedway + GDC Camera + LIDAR-based pedestrian datasets
- (Stanford Drone Dataset)
- (ETH-UCY Dataset)
- (MOT Dataset)

Data processing consists of the following high-level steps:

- For each source dataset
  - Extract trajectory data into a graph format consisting of:
    - Top-down estimates of:
        - Pedestrian position and orientation (x, y, theta) at each timestep
        - Pedestrian velocity (vx, vy) at each timestep
        - Pedestrian acceleration (ax, ay) at each timestep
        - Pedestrian heading (theta) at each timestep
        - Pedestrian bounding box (x, y, width, height) at each timestep
    - Image features:
        - SAM3 segmentation of each visible agent

## Data Format

TBD
