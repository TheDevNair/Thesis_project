import uproot
import awkward as ak
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
import shutil
import time

# Configuration
INPUT_DIR = r"C:\Users\Dev\OneDrive\Desktop\hbb_data_backup"
OUTPUT_DIR = r"C:\Users\Dev\OneDrive\Desktop\hbb_production_parquet"
TREE_NAME = "deepntuplizer/tree"

def get_arrow_type(arr: ak.Array) -> pa.DataType:
    """Optimal type conversion with physics-data validation"""
    layout = arr.layout
    
    if isinstance(layout, ak.contents.ListOffsetArray):
        content_type = get_arrow_type(ak.Array(layout.content))
        return pa.list_(content_type)
    
    if isinstance(layout, ak.contents.NumpyArray):
        dtype = layout.dtype
        
        # Enhanced integer handling
        if dtype == np.int32:
            min_val, max_val = ak.min(arr), ak.max(arr)
            if np.iinfo(np.int16).min < min_val < max_val < np.iinfo(np.int16).max:
                return pa.int16()
            return pa.int32()
        
        # Precision-safe float conversion
        if dtype == np.float64:
            f32_arr = arr.astype(np.float32)
            if ak.all(ak.isclose(arr, f32_arr, rtol=1e-5, atol=1e-5)):
                return pa.float32()
            return pa.float64()
        
        return pa.from_numpy_dtype(dtype)
    
    raise TypeError(f"Unsupported type: {type(layout)}")

def convert_root_to_parquet_production():
    """Production-ready converter with physics-data preservation"""
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    root_files = list(input_path.glob("*.root"))
    print(f"🚚 Found {len(root_files)} ROOT files for conversion")

    for root_file in root_files:
        file_start = time.time()
        print(f"\n📦 Processing: {root_file.name}")

        try:
            with uproot.open(root_file) as file:
                tree = file[TREE_NAME]
                awkward_array = tree.arrays(library="ak")

                schema = []
                arrays = []
                for field in awkward_array.fields:
                    arr = awkward_array[field]
                    original_dtype = arr.layout.dtype if hasattr(arr.layout, 'dtype') else None
                    
                    try:
                        pa_type = get_arrow_type(arr)
                        arrow_arr = ak.to_arrow(arr, extensionarray=False)
                        arrow_arr = arrow_arr.cast(pa_type) if arrow_arr.type != pa_type else arrow_arr
                    except Exception as e:
                        print(f"⚠️ Type conversion failed for {field}: {str(e)}")
                        print(f"   Falling back to original type: {original_dtype}")
                        pa_type = pa.from_numpy_dtype(original_dtype)
                        arrow_arr = ak.to_arrow(arr, extensionarray=False)

                    # Ensure field name is a string and type is valid
                    schema.append(pa.field(str(field), pa_type))  # Convert field name to string
                    arrays.append(arrow_arr)

                # Create schema explicitly
                arrow_schema = pa.schema(schema)
                
                # Write with optimized settings
                output_file = output_path / f"{root_file.stem}.parquet"
                pq.write_table(
                    pa.Table.from_arrays(arrays, schema=arrow_schema),
                    output_file,
                    compression="ZSTD",
                    compression_level=15,
                    use_dictionary=True,
                    row_group_size=500000,
                    data_page_size=1 << 20,
                    write_statistics=True
                )

                # Compression report
                orig_size = root_file.stat().st_size / 1e6
                parq_size = output_file.stat().st_size / 1e6
                print(f"✅ {parq_size:.1f}MB (from {orig_size:.1f}MB)")
                print(f"⏱️ {time.time()-file_start:.1f}s | 📉 Ratio: {parq_size/orig_size:.2f}x")

        except Exception as e:
            print(f"🔥 Critical failure: {str(e)}")
            continue

if __name__ == "__main__":
    total_start = time.time()
    convert_root_to_parquet_production()
    print(f"\n🏁 Total conversion time: {time.time()-total_start:.1f}s")