#!/usr/bin/env python3
import argparse
import time
import zmq

# Demo sequence driver for Alice and Bob controller epy blocks
# Sends short text commands as ZMQ PUB messages. The GRC graphs have
# zeromq_sub_msg_source with bind=False (they connect), so this driver binds.

CMDS_ALICE = [
    ("start_tx", 1.0),
    ("collect", 1.0),
    ("quantized", 0.5),
    ("parity_sent", 0.5),
    ("privacy_done", 0.5),
]

CMDS_BOB_OK = [
    ("collect", 1.0),
    ("parity_recv", 0.5),
    ("reconcile_ok", 0.5),
    ("privacy_done", 0.5),
]

CMDS_BOB_FAIL = [
    ("collect", 1.0),
    ("parity_recv", 0.5),
    ("reconcile_fail", 0.5),
]


def main():
    ap = argparse.ArgumentParser(description="PHYSEC demo controller driver")
    ap.add_argument("--alice_ctrl", default="tcp://*:9002", help="Bind address for Alice control PUB (SUBs connect to this)")
    ap.add_argument("--bob_ctrl", default="tcp://*:9001", help="Bind address for Bob control PUB (SUBs connect to this)")
    ap.add_argument("--bob_outcome", choices=["ok", "fail"], default="ok", help="Reconciliation result to signal to Bob")
    ap.add_argument("--pace", type=float, default=0.5, help="Additional delay between commands (s)")
    args = ap.parse_args()

    ctx = zmq.Context.instance()

    # Bind PUBs
    alice_pub = ctx.socket(zmq.PUB)
    alice_pub.bind(args.alice_ctrl)

    bob_pub = ctx.socket(zmq.PUB)
    bob_pub.bind(args.bob_ctrl)

    # Give subscribers time to connect
    time.sleep(1.0)

    def send(sock, label, cmd):
        print(f"[{label}] -> {cmd}")
        sock.send_string(cmd)

    # Alice sequence
    for cmd, wait_s in CMDS_ALICE:
        send(alice_pub, "ALICE", cmd)
        time.sleep(wait_s + args.pace)

    # Bob sequence
    cmds_bob = CMDS_BOB_OK if args.bob_outcome == "ok" else CMDS_BOB_FAIL
    for cmd, wait_s in cmds_bob:
        send(bob_pub, "BOB", cmd)
        time.sleep(wait_s + args.pace)

    print("Demo sequence completed.")


if __name__ == "__main__":
    main()
