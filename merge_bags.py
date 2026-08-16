#!/usr/bin/env python3
"""Merge multiple ros2 bags into one time-ordered bag.

Usage:
  python3 merge_bags.py --out /bags/merged \
      --bag /bags/a --bag /bags/b:robot_2 --bag /bags/c:robot_3

A ":prefix" suffix on a --bag remaps that bag's topics that aren't already
namespaced under /<prefix>/ to /<prefix>/<topic> so identical topic names
from different robots don't collide. TF and other global topics are never
remapped. Message timestamps are preserved (bags recorded on a shared wall
clock stay aligned).
"""
import argparse
import heapq

import rosbag2_py

NEVER_REMAP = {"/tf", "/tf_static", "/clock", "/rosout", "/parameter_events"}


def open_reader(path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    return reader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bag", action="append", required=True,
                    help="bag path, optionally PATH:prefix")
    args = ap.parse_args()

    bags = []
    for spec in args.bag:
        path, _, prefix = spec.partition(":")
        bags.append((path, prefix or None))

    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=args.out, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))

    readers = []
    registered = {}
    for path, prefix in bags:
        reader = open_reader(path)
        remap = {}
        for t in reader.get_all_topics_and_types():
            new_name = t.name
            if (prefix and t.name not in NEVER_REMAP
                    and not t.name.startswith(f"/{prefix}/")):
                new_name = f"/{prefix}" + t.name
            remap[t.name] = new_name
            if new_name not in registered:
                writer.create_topic(rosbag2_py.TopicMetadata(
                    name=new_name, type=t.type,
                    serialization_format=t.serialization_format,
                    offered_qos_profiles=t.offered_qos_profiles))
                registered[new_name] = t.type
                print(f"  topic: {t.name} -> {new_name} ({t.type})")
            elif registered[new_name] != t.type:
                raise RuntimeError(
                    f"type clash on {new_name}: {registered[new_name]} vs "
                    f"{t.type}")
        readers.append((reader, remap))

    heap = []
    for i, (reader, _remap) in enumerate(readers):
        if reader.has_next():
            topic, data, t = reader.read_next()
            heapq.heappush(heap, (t, i, topic, data))

    count = 0
    while heap:
        t, i, topic, data = heapq.heappop(heap)
        reader, remap = readers[i]
        writer.write(remap[topic], data, t)
        count += 1
        if count % 20000 == 0:
            print(f"  ...{count} messages written", flush=True)
        if reader.has_next():
            ntopic, ndata, nt = reader.read_next()
            heapq.heappush(heap, (nt, i, ntopic, ndata))

    print(f"Done. Wrote {count} messages to {args.out}")


if __name__ == "__main__":
    main()
