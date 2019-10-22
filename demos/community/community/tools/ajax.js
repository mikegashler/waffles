
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
	alert("The back-end server replied: " + response);

	// Parse the JSON
	let ob = JSON.parse(response);

	alert("someval = " + ob.someval);
}

let ob = {};
ob.burrito = 1.2345;
let str = JSON.stringify(ob);
httpPost("/a", str, cb);
