import imas_west

def get_equilibrium(nshot):
    return imas_west.get(nshot, 'equilibrium', 0, 1)



import sys
import pickle
if __name__ == "__main__":
    # try:
    #     kdhfz
    # except Exception as e:
    #     raise RuntimeError(f"Failed to deserialize input: {sys.executable}")
    
    # # Debug: Check if input data is received
    f = open("demofile2.txt", "a")
    f.write("New try")
    f.write(sys.version)
    f.write(sys.executable)
    f.close()
    raw_input = sys.stdin.buffer.read()
    if not raw_input:
        raise RuntimeError("No input received by subprocess.")
    print("Raw Input Received.")
    # Deserialize input data
    try:
        input_data = pickle.loads(raw_input)
    except Exception as e:
        raise RuntimeError(f"Failed to deserialize input: {e}")

    func_name = input_data["func_name"]
    args = [deserialize_data(arg) for arg in input_data["args"]]
    # sys.stdout.buffer.write(pickle.dumps(args))

    if func_name in globals() and callable(globals()[func_name]):
        result = globals()[func_name](*args)
        sys.stdout.buffer.write(pickle.dumps(result))
    else:
        sys.stdout.buffer.write(pickle.dumps({"error": f"Function {func_name} not found"}))



def deserialize_data(data):
    if data["type"] == "sparse":
        return pickle.loads(data["data"])
    elif data["type"] == "ndarray":
        return data["data"]
    elif data["type"] == "primitive":
        return data["data"]
    elif data["type"] == "list":
        return data["data"]
    elif data["type"] == "dict":
        return data["data"]
    else:
        raise ValueError(f"Unsupported data type: {data['type']}")