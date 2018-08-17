function httpPost(url, payload, callback)
{
	let request = new XMLHttpRequest();
	request.onreadystatechange = function()
	{
		if(request.readyState == 4)
		{
			if(request.status == 200)
			callback(request.responseText);
			else
			{
				if(request.status == 0 && request.statusText.length == 0)
					alert("Connection failed");
				else
					alert("Server returned status " + request.status + ", " + request.statusText);
			}
		}
	};
	request.open('post', url, true);
	request.setRequestHeader('Content-Type',
	'application/x-www-form-urlencoded');
	request.send(payload);
}

function setListContents(listName, listContents, parent)
{
	let selectBox = document.getElementById(listName);
	selectBox.options.length = 0;
	if(parent)
	{
		let opt = document.createElement("option");
		opt.text = ".. (Parent folder)";
		selectBox.add(opt);
	}
	for(let i = 0; i < listContents.length; i++)
	{
		let opt = document.createElement("option");
		opt.text = listContents[i];
		selectBox.add(opt);
	}
}

function cb(response)
{
	//alert("received response: " + response);
	let ob = JSON.parse(response);
	let pathBox = document.getElementById("path");
	pathBox.value = ob.path + "/";
	setListContents("folders", ob.folders, ob.path.length > 0 ? true : false);
	setListContents("files", ob.files, false);
}

function closeDetails(except)
{
	if(except != 1) document.getElementById("d1").removeAttribute("open");
	if(except != 2) document.getElementById("d2").removeAttribute("open");
	if(except != 3) document.getElementById("d3").removeAttribute("open");
	if(except != 4) document.getElementById("d4").removeAttribute("open");
}

function onclickd1()
{
	let selectBox = document.getElementById("folders");
	if(selectBox.value.length < 1)
	{
		alert("nothing selected");
		setTimeout(function() { closeDetails(0); }, 50);
	}
	closeDetails(1);
}

function onclickd2()
{
	closeDetails(2);
}

function onclickd3()
{
	let selectBox = document.getElementById("files");
	if(selectBox.value.length < 1)
	{
		alert("nothing selected");
		setTimeout(function() { closeDetails(0); }, 50);
	}
	closeDetails(3);
}

function onclickd4()
{
	closeDetails(4);
}

function onfolderchange()
{
	let selectBox = document.getElementById("folders");
	let ob = {};
	ob.action = "filelist";
	ob.folder = selectBox.value;
	httpPost("/ajax", JSON.stringify(ob), cb);
}

function editgui()
{
	let selectBox = document.getElementById("files");
	window.location = "/edit?pagename=" + selectBox.value;
}

function edittext()
{
	let selectBox = document.getElementById("files");
	window.location = "/edittext?pagename=" + selectBox.value;
}

function newfolder()
{
	closeDetails();
	alert("Sorry, not implemented yet");
}

function onPageLoad()
{
	let ob = {};
	ob.action = "filelist";
	ob.folder = ".";
	httpPost("/ajax", JSON.stringify(ob), cb);
}

onPageLoad();