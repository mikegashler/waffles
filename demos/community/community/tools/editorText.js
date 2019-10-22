
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

function cb(response)
{
	alert("received response: " + response);
}

function save()
{
	let ob = {};
	ob.action = "save_text";
	ob.page = document.getElementById("page").value;
	ob.filename = document.getElementById("filename").value;
	httpPost("/a", JSON.stringify(ob), cb);
}
