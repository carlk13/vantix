use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray4, PyArrayMethods};
use rayon::prelude::*;
use std::path::Path;
use fast_image_resize::{ResizeAlg, Resizer, ResizeOptions, images::Image as FastImage, PixelType};
use rand::prelude::*;


#[pyfunction]
fn load_images_fast<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    width: u32,
    height: u32,
    augment: bool 
) -> PyResult<Bound<'py, PyArray4<f32>>> {

    let batch_size = paths.len();
    let area = (width * height) as usize;
    let total_floats = batch_size * 3 * area;

    // Allocate memory 
    let mut buffer = vec![0.0f32; total_floats];

    // Release GIL 
    {
        let buffer_slice = buffer.as_mut_slice();
        let paths_slice = &paths;

        py.allow_threads(move || {
            buffer_slice.par_chunks_exact_mut(3 * area)
                .zip(paths_slice.par_iter())
                .for_each(|(chunk, path)| {
                    process_single_image(path, width, height, augment, chunk, area);
                });
        });
    }

    // Re-acquire GIL to create NumPy array
    let array_1d = buffer.into_pyarray(py);
    let reshaped = array_1d.reshape([batch_size, 3, height as usize, width as usize])?;

    Ok(reshaped)
}


fn process_single_image(
    path: &str, 
    target_w: u32, 
    target_h: u32, 
    augment: bool, 
    out_buf: &mut [f32],
    area: usize
) {
    let img = match image::open(Path::new(path)) {
        Ok(i) => i,
        Err(_) => return, 
    };

    let src_width = img.width();
    let src_height = img.height();
    let src_data = img.to_rgb8().into_raw(); // Consumes the buffer into Vec<u8>

    let src_image = FastImage::from_vec_u8(
        src_width,
        src_height,
        src_data,
        PixelType::U8x3,
    ).unwrap();

    let mut dst_image = FastImage::new(target_w, target_h, PixelType::U8x3);
    
    let mut resizer = Resizer::new();

    let options = ResizeOptions::new()
        .resize_alg(ResizeAlg::Convolution(fast_image_resize::FilterType::Bilinear));

    if resizer.resize(&src_image, &mut dst_image, &options).is_ok() {
        let data = dst_image.buffer();

        let mut do_flip = false;
        if augment {
            let mut rng = rand::thread_rng();
            do_flip = rng.gen_bool(0.5);
        }

        let inv_255 = 1.0 / 255.0;
        let stride = (target_w * 3) as usize;

        for y in 0..target_h {
            for x in 0..target_w {
                let src_x = if do_flip { target_w - 1 - x } else { x };
                
                let src_idx = (y as usize * stride) + (src_x as usize * 3);
                
                let pixel_r = data[src_idx];
                let pixel_g = data[src_idx + 1];
                let pixel_b = data[src_idx + 2];

                let flat_idx = (y as usize * target_w as usize) + x as usize;

                out_buf[flat_idx]          = pixel_r as f32 * inv_255; // R
                out_buf[flat_idx + area]   = pixel_g as f32 * inv_255; // G
                out_buf[flat_idx + 2*area] = pixel_b as f32 * inv_255; // B
            }
        }
    }
}

#[pymodule]
fn _vantix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_images_fast, m)?)?;
    Ok(())
}