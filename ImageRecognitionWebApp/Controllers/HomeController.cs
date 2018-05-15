using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Web;
using System.Web.Mvc;

namespace ImageRecognitionWebApp.Controllers
{
    public class HomeController : Controller
    {
        public ActionResult Index()
        {
            return View();
        }

        public JsonResult LoadImages()
        {
            string[] filePaths = Directory.GetFiles("D:\\Projects\\ImageRecognition\\ImageRecognition\\ImageRecognition\\images");

            List<JsonResult> imageJson = new List<JsonResult>();

            Dictionary<string, List<JsonResult>> tmp = new Dictionary<string, List<JsonResult>>();
            tmp.Add("C1", new List<JsonResult>());

            foreach (string filePath in filePaths)
            {
                switch (Path.GetFileName(filePath)[0])
                {
                    case '1':
                        imageJson.Add(CreateImageForDisplay(filePaths.Where(x => (Path.GetFileName(x)[0] == '1')).ToArray(), "C1"));
                        break;
                        //case '2':
                        //    imageJson.Add(CreateImageForDisplay(filePath, "A1"));
                        //    break;
                        //case '3':
                        //    imageJson.Add(CreateImageForDisplay(filePath, "C2"));
                        //    break;
                        //case '4':
                        //    imageJson.Add(CreateImageForDisplay(filePath, "A2"));
                        //    break;
                        //case '5':
                        //    imageJson.Add(CreateImageForDisplay(filePath, "MP"));
                        //    break;
                        //case '6':
                        //    imageJson.Add(CreateImageForDisplay(filePath, "DP"));
                        
                    default:
                        break;
                }
            }

            return Json(imageJson, JsonRequestBehavior.AllowGet);
        }

        public JsonResult CreateImageForDisplay(string[] filePath, string layer)
        {
            var bitmap = new Bitmap(1133, 309);

            var heightX = 0;
            var widthY = 0;

            for (int i = 0; i < filePath.Length; i++)
            {
                Image bmp = Image.FromFile(filePath[i]);

                Bitmap resizedImage = ResizeImage(bmp, 100, 100);

                using (var canvas = Graphics.FromImage(bitmap))
                {
                    canvas.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    //Draw each image (maybe use a loop to loop over images to draw)

                    if (heightX > 1000)
                    {
                        heightX = 0;
                        widthY += 103;
                    }

                    canvas.DrawImage(resizedImage, new Rectangle(0, 0, widthY, heightX), new Rectangle(0, 0, widthY, heightX), GraphicsUnit.Pixel);

                    heightX += 103;

                    canvas.Save();

                    bitmap.Save("D://test1.png", ImageFormat.Png);
                }
            }



            bitmap.Save("D://test.png", ImageFormat.Png);

            //var stream = new MemoryStream();

            //resizedImage.Save(stream, ImageFormat.Png);

            //var resizedImageArray = stream.ToArray();

            //var resizedImageBase64 = Convert.ToBase64String(resizedImageArray);

            return Json(new { layer = "data:image/png;base64,"/* + resizedImageBase64 */}, JsonRequestBehavior.AllowGet);
        }

        public JsonResult CreateImage(List<string[]> slika)
        {
            List<double> polje = new List<double>();

            foreach (var item in slika) //item je polje
            {
                foreach (var element in item) // element je broj
                {
                    var pixel = Math.Round(double.Parse(element) * 255);

                    polje.Add(pixel);
                }
            }

            var min = polje.Min();
            var max = polje.Max();
            var range = max - min;

            polje = polje.Select(d => (d - min) / range).Select(n => ((1 - n) * 0 + n * 255)).ToList();


            Bitmap newBitmap = new Bitmap(26, 26);

            for (int j = 0; j < 26; j++)
            {
                for (int i = 0; i < 26; i++)
                {
                    Color newColor = Color.FromArgb(Convert.ToInt32(polje[i + j * 26]), Convert.ToInt32(polje[i + j * 26]), Convert.ToInt32(polje[i + j * 26]));

                    newBitmap.SetPixel(i, j, newColor);
                }
            }

            Image img = (Image)newBitmap;

            img = ResizeImage(img, 50, 50);

            img.Save("D:\\orginal.png");

            Bitmap bmp = new Bitmap(img);

            //bmp = InvertColors(bmp);

            //bmp.Save("D:\\okrenuta.png");

            var stream = new MemoryStream();

            bmp.Save(stream, ImageFormat.Png);

            var resizedImageArray = stream.ToArray();

            var resizedImageBase64 = Convert.ToBase64String(resizedImageArray);

            var json = Json("data:image/png;base64," + resizedImageBase64);

            return json;
        }

        public JsonResult SaveImage(string slika)
        {
            var polje = slika.Split(',');

            var image = polje[1];

            var imageByte = Convert.FromBase64String(image);

            var stream = new MemoryStream(imageByte, 0, imageByte.Length);

            var img = Image.FromStream(stream, true); //tu je bio image

            //Bitmap resizedImage = ResizeImage(img, 28, 28);

            Bitmap resizedImage = new Bitmap(img);

            resizedImage = InvertColors(resizedImage);

            //resizedImage = ClearAllPixels(resizedImage);

            resizedImage.Save(Server.MapPath("~/App_data/NacrtanaSlika.png"));

            //resizedImage = InvertColors(resizedImage);

            //resizedImage.Save(Server.MapPath("~/App_data/InvertColour.png"));

            stream = new MemoryStream();

            resizedImage.Save(stream, ImageFormat.Png);

            var resizedImageArray = stream.ToArray();

            var resizedImageBase64 = Convert.ToBase64String(resizedImageArray);

            var json = Json(polje[0] + "," + resizedImageBase64);

            return json;

        }

        private Bitmap InvertColors(Bitmap resizedImage)
        {
            Bitmap pic = new Bitmap(resizedImage);

            for (int y = 0; (y <= (pic.Height - 1)); y++)
            {
                for (int x = 0; (x <= (pic.Width - 1)); x++)
                {
                    Color inv = pic.GetPixel(x, y);

                    //System.Diagnostics.Debug.WriteLine(pic.GetPixel(x, y));

                    inv = Color.FromArgb(255, (255 - inv.R), (255 - inv.G), (255 - inv.B));

                    //inv = Color.FromArgb(inv.ToArgb() ^ 0xffffff);

                    pic.SetPixel(x, y, inv);
                }
            }

            return pic;
        }

        private Bitmap ClearAllPixels(Bitmap resizedImage)
        {
            Bitmap newImage = new Bitmap(resizedImage.Width, resizedImage.Height);

            Color newColor = Color.White;

            for (int i = 0; i < resizedImage.Width; i++)
            {
                for (int j = 0; j < resizedImage.Height; j++)
                {
                    var pixel = resizedImage.GetPixel(i, j);

                    //System.Diagnostics.Debug.WriteLine($"{pixel.A} {pixel.R} {pixel.G} {pixel.B}");

                    if (pixel.A != 255)
                    {
                        newImage.SetPixel(i, j, newColor);
                    }
                    else
                    {
                        newImage.SetPixel(i, j, pixel);
                    }


                }
            }

            return newImage;
        }

        public static Bitmap ResizeImage(Image image, int width, int height) //tu je bio image
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.Default;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;



                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }
    }
}