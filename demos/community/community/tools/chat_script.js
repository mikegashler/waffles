let comments_file = "users/test_user/comments.json";
let server_addr = "http://gashler.com:8988/a";

function tog_viz(id)
{
	var x = document.getElementById(id);
  if (x.style.display == "block") {
		x.style.display = "none";
  } else {
		x.style.display = "block";
		var tb = document.getElementById(id + 't');
		tb.focus();
  }
}

let cookie = null;

function ajax(url, outgoing, callback)
{
	// Add a cookie to the object, since some browsers block HTTP header cookies in AJAX
	let c = document.cookie;
	if(!c || c.indexOf("GDPSI=") < 0)
		c = cookie;
	if(c)
		outgoing.cookie = c;
	let payload = JSON.stringify(outgoing);
	let request = new XMLHttpRequest();
	request.onreadystatechange = function() {
		if(request.readyState == 4) {
			if(request.status == 200) {
				//console.log("The server replied: " + request.responseText);
				let incoming = JSON.parse(request.responseText);
				if(incoming.cookie) {
					cookie = incoming.cookie;
					delete incoming.cookie;
				}
				callback(incoming);
			}	else {
				if(request.status == 0 && request.statusText.length == 0)
					console.log("Connection failed");
				else
					console.log("Server returned status " + request.status + ", " + request.statusText);
			}
		}
	};
	request.open('post', url, true);
	request.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
	request.send(payload);
	//console.log("Sent message: " + payload);
}

function refresh_comments_callback(incoming) {
	let comments = document.getElementById("comments");
	comments.innerHTML = incoming.html;
}

function refresh_comments() {
	// Find the text field
	let comments = document.getElementById("comments");

	// Make a JSON blob
	let outgoing = {};
	outgoing.action = "get_comments";
	outgoing.file = comments_file;

	// Send the JSON blob to the server
	ajax(server_addr, outgoing, refresh_comments_callback);
}

function submit_comment_callback(incoming) {
	refresh_comments();
}

function post_comment(id) {
	// Find the text field
	let msg = document.getElementById(id);

	// Make a JSON blob
	let ob = {};
	ob.action = "add_comment";
	ob.file = comments_file;
	ob.id = id;
	ob.comment = msg.value;

	// Send the JSON blob to the server
	ajax(server_addr, ob, submit_comment_callback);
}

refresh_comments();
