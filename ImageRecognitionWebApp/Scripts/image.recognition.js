window.onload = function () {

	createCanvas();

    $("#canvasDisplayDiv").hide();

    $("#resultDiv").hide();

    $("#displayDiv").hide();
        
    $("#btnRecognize").on('click', function () {

        $("#canvasDisplayDiv").show();

        var result = saveImage();

        $("#tblResults").show();

    });

    $("#btnClear").on('click', function () {

        clearCanvas();

    });

	
}

function createCanvas() {


	var canvasDiv = document.getElementById('canvasDiv');

	canvas = document.createElement('canvas');

	canvas.setAttribute('width', 150);
	canvas.setAttribute('height', 150);
	canvas.style.backgroundColor = 'white'; //bio je white
	canvas.style.border = '2px solid';
    canvas.style.margin = '10px';
    canvas.style.marginLeft  = 'auto';
    canvas.style.marginRight = 'auto';
    canvas.style.display = 'block';



	canvas.setAttribute('id', 'canvas');

	canvasDiv.appendChild(canvas);

	if (typeof G_vmlCanvasManager != 'undefined') {
		canvas = G_vmlCanvasManager.initElement(canvas);
	}

    var context = canvas.getContext("2d");

    context.fillStyle = '#fff';  /// set white fill style
    context.fillRect(0, 0, canvas.width, canvas.height);

	$('#canvas').mousedown(function (e) {

		var mouseX = e.pageX - this.offsetLeft;
		var mouseY = e.pageY - this.offsetTop;

		paint = true;
		addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop);
		redraw();

	});

	$('#canvas').mousemove(function (e) {
		if (paint) {
			addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true);
			redraw();
		}
	});

	$('#canvas').mouseup(function (e) {
		paint = false;
	});

	$('#canvas').mouseleave(function (e) {
		paint = false;
	});

	var clickX = new Array();
	var clickY = new Array();
	var clickDrag = new Array();
	var paint;

	function addClick(x, y, dragging) {
		clickX.push(x);
		clickY.push(y);
		clickDrag.push(dragging);
	}

	function redraw() {
		context.clearRect(0, 0, context.canvas.width, context.canvas.height); // Clears the canvas

		context.strokeStyle = "black"; //tu je bilo black
		context.lineJoin = "round";
		context.lineWidth = 15;

		for (var i = 0; i < clickX.length; i++) {
			context.beginPath();
			if (clickDrag[i] && i) {
				context.moveTo(clickX[i - 1], clickY[i - 1]);
			} else {
				context.moveTo(clickX[i] - 1, clickY[i]);
			}
			context.lineTo(clickX[i], clickY[i]);
			context.closePath();
			context.stroke();
		}
	}


}

function clearCanvas() {

	var canvas = document.getElementById('canvas');
	canvas.outerHTML = "";
	delete canvas;
	createCanvas();

	var pictureCanvas = document.getElementById('pictureCanvas');
	var context = pictureCanvas.getContext("2d");

	context.clearRect(0, 0, pictureCanvas.width, pictureCanvas.height);

    $("#firstPred").empty();
    $("#firstAcc").empty();
    $("#secondPred").empty();
    $("#secondAcc").empty();
    $("#thirdPred").empty();
    $("#thirdAcc").empty();

    $("#resultDiv").hide();
    $("#canvasDisplayDiv").hide();

    $("#displayDiv").html('');
    $("#displayDiv").hide();
}

function saveImage() {    

    var canvas = document.getElementById('canvas');
    var context = canvas.getContext('2d');

    var imageData = context.getImageData(0, 0, 150, 150);
    var compositeOperation = context.globalCompositeOperation;
    context.globalCompositeOperation = "destination-over";
    context.fillStyle = 'white';
    context.fillRect(0, 0, 150, 150);

    var slika = canvas.toDataURL();
            
	$.ajax({
		type: "POST",
		data: JSON.stringify({ "slika": slika }),
		url: "../Home/SaveImage",
		contentType: 'application/json',
		success: function (data) {
			console.log("Success:");
			console.log(data);
			showImageInCanvas(data);
		},
		error: function (error) {
            console.log("Error:");
            console.log(error);
		}
	});
}

function showImageInCanvas(data) {

	var canvas = document.getElementById('pictureCanvas');
    var context = canvas.getContext("2d"); 
    	
    var image = new Image();

    var imgWidth = 28 / 2;
    var imgHeight = 28 / 2;

	image.onload = function () {
        //context.drawImage(image, canvas.width / 2 - image.width / 2, canvas.height / 2 - image.height / 2,28,28);
        context.drawImage(image, canvas.width / 2 - imgWidth, canvas.height / 2 - imgHeight, 28, 28);
	}

    image.src = data;
       

	//Tu sam posatvio malu sliku da se vidi kaj je otišlo prema mreži

	//Tu šaljem sliku u mrežu

    var arr = data.split(',');

	var newDataString = 'data:image/png;base64,' + arr[1];//jpeg

    console.log('showImageInCanvas -> ' + newDataString);

    getPrediction(newDataString);


}

function getPrediction(data) {

	$.ajax({
		type: "POST",
		data: JSON.stringify({ "slika": data }),
		url: "http://localhost:5000/api/GetPrediction",
		contentType: 'application/json',
		success: function (data) {
			console.log("Success:");
			console.log(data);
			displayData(data);
		},
		error: function (error) {
			console.log("Error:")
			console.log(error);
		}
	});

}

function displayData(data) {

    var firstPred = data["results"][0]["key"];
    var firstAcc = parseFloat(data["results"][0]["value"]).toFixed(2) * 100;

    $("#firstPred").append(firstPred);
    $("#firstAcc").append(firstAcc + "%");

    var secondPred = data["results"][1]["key"];
    var secondAcc = parseFloat(data["results"][1]["value"]).toFixed(2) * 100;

    $("#secondPred").append(secondPred);
    $("#secondAcc").append(secondAcc + "%");

    var thirdPred = data["results"][2]["key"];
    var thirdAcc = parseFloat(data["results"][2]["value"]).toFixed(2) * 100;

    $("#thirdPred").append(thirdPred);
    $("#thirdAcc").append(thirdAcc + "%");

    $("#resultDiv").show();
    
    displayLayerImages(data["images"]);

}

function displayLayerImages(data) {

    for (var i = 0; i < data.length; i++) {

        var label = document.createElement("label");
        label.innerHTML = data[i]["name"];

        $("#displayDiv").append(label);

        var img = document.createElement("img");
        img.src = "data:image/png;base64," + data[i]["picture"];
        img.style.width = "100%";

        $("#displayDiv").append(img);

    }

    $("#displayDiv").show();


    $.ajax({
        type: "GET",
        //data: JSON.stringify({ "slika": data }),
        url: "http://localhost:5000/api/GetImages",
        contentType: 'image/png',
        success: function (data) {
            console.log("Success GetImages:");
            //Test(data);            
        },
        error: function (error) {
            console.log("Error GetImages:")
            console.log(error);
        }
    });


    ////$("#conv2dImg").attr('crossOrigin', 'anonymous');
    ////$("#conv2dImg").attr('src', "/Images/Conv2d_1.png");

    //$("#conv2dImg").on('bestfit', function () {
    //    var css;
    //    var ratio = $(this).width() / $(this).height();
    //    var pratio = $(this).parent().width() / $(this).parent().height();
    //    if (ratio < pratio) css = { width: 'auto', height: '100%' };
    //    else css = { width: '100%', height: 'auto' };
    //    $(this).css(css);
    //}).on('load', function () {
    //    $(this).trigger('bestfit');
    //}).trigger('bestfit');
}

function Test(data) {
       
    var x = "data:image/png;base64," + data["images"];
    
    var label = document.createElement("label");
    label.innerHTML = "Slika poslana preko flaska";

    $("#displayDiv").append(label);

    var img = document.createElement("img");
    img.src = x;
    img.style.width = "100%";

    $("#displayDiv").append(img);
    

}







