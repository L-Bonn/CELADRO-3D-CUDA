import struct

def read_cellHist_binary(filename):
    """
    Reads snapshots from 'filename' in the order:
      [timeVal (double, 8 bytes)]
      [nCells  (int,    4 bytes)]
      For each of the nCells (36 bytes):
         cellID       (int,    4 bytes)
         birth_time   (double, 8 bytes)
         death_time   (double, 8 bytes)
         parent       (int,    4 bytes)
         physicalprop (double, 8 bytes)
         generation   (int,    4 bytes)

    Returns a list of snapshots. Each snapshot is (timeVal, cellMap).
    'cellMap' is a dict: { cellID -> {
        "birth_time":   float,
        "death_time":   float,
        "parent":       int,
        "physicalprop": float,
        "generation":   int
    } }
    """
    snapshots = []
    with open(filename, 'rb') as f:
        while True:
            # 1) Read timeVal (double = 8 bytes)
            buf = f.read(8)
            if len(buf) < 8:
                break  # end-of-file
            (timeVal,) = struct.unpack('=d', buf)

            # 2) Read nCells (int = 4 bytes)
            buf = f.read(4)
            if len(buf) < 4:
                break
            (nCells,) = struct.unpack('=i', buf)

            cellMap = {}
            for _ in range(nCells):
                # Read the 36-byte chunk for each cell
                chunk = f.read(36)
                if len(chunk) < 36:
                    break  # partial data => break

                # Unpack in the same order we wrote:
                # i, d, d, i, d, i  =>  4 + 8 + 8 + 4 + 8 + 4 = 36
                cellID, birth_time, death_time, parent, physicalprop, generation = \
                    struct.unpack('=i d d i d i', chunk)

                # Store into a Python dict
                cellMap[cellID] = {
                    "birth_time":   birth_time,
                    "death_time":   death_time,
                    "parent":       parent,
                    "physicalprop": physicalprop,
                    "generation":   generation
                }

            snapshots.append((timeVal, cellMap))

    return snapshots


if __name__ == "__main__":
    result = read_cellHist_binary("cellHist.bin")
    for (t, cmap) in result:
        print(f"\nSnapshot at timeVal={t}, nCells={len(cmap)}")
        for cid, info in cmap.items():
            print(f"  Cell {cid}: birth={info['birth_time']}, "
                  f"death={info['death_time']}, parent={info['parent']}, "
                  f"prop={info['physicalprop']}, gen={info['generation']}")

