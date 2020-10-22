import java.awt.image.*; 
import java.io.*;
import javax.imageio.*;

public class ConvertRGBToPNG {
	final int width = 352;
	final int height = 288;
	BufferedImage image;

	private void readImageRGB(String imgPath, BufferedImage img) {
		try {
			int frameLength = width * height * 3;

			File file = new File(imgPath);
			RandomAccessFile raf = new RandomAccessFile(file, "r");
			raf.seek(0);

			long len = frameLength;
			byte[] bytes = new byte[(int) len];

			raf.read(bytes);

			int ind = 0;
			for(int y = 0; y < height; y++)
			{
				for(int x = 0; x < width; x++)
				{
					byte r = bytes[ind];
					byte g = bytes[ind+height*width];
					byte b = bytes[ind+height*width*2]; 

					int pix = 0xff000000 | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
					img.setRGB(x,y,pix);
					ind++;
				}
			}
			raf.close();
		} 
		catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void writeFile(String outFileDir, BufferedImage img, String index) {
		try {
			String filePath= outFileDir + "/image-" + index + ".png";
			File outFile = new File(filePath);
			ImageIO.write(img, "PNG", outFile);
			System.out.println("Writing file: image-" + index + ".png");
		}
		catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void convert(String[] args) {
		String inputDir = args[0];
		String outputDir = args[1];

		String[] imagePathnames; 
		File f = new File(inputDir); 
		imagePathnames = f.list();
		for (String imagePathname : imagePathnames) {
			String index = imagePathname.split("-")[1];
			index = index.substring(0, 4);

			image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
			readImageRGB(inputDir + "/" + imagePathname, image);
			writeFile(outputDir, image, index);
		}
	}

	public static void main(String[] args) {
		ConvertRGBToPNG converter = new ConvertRGBToPNG();
		converter.convert(args);
	}
}
