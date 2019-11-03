
function firstchild(node)
{
	for(let i = 0; i < node.childNodes.length; i++)
	{
		let c = node.childNodes[i];
		if(c.nodeName === "#text" && c.length == 0)
			continue;
		return c;
	}
	return null;
}

function remove(condemned)
{
	let range = window.getSelection().getRangeAt(0);
	let node = range.commonAncestorContainer;
	if(node.childNodes.length > 0)
	{
		if(node.nodeName === condemned)
			node = firstchild(node);
		else
		{
			let fc = firstchild(node);
			if(fc !== null && fc.nodeName === condemned)
				node = fc;
		}
		if(node.nodeName === condemned && firstchild(node) !== null)
			node = firstchild(node);
	}
	let i = 0;
	while(true)
	{
		let par = node.parentNode;
		if(par == null)
			break;
		if(par.nodeName === condemned)
		{
			// Attach all the grand-children to their grand-parent
			for(let i = 0; i < par.childNodes.length; i++)
				par.parentNode.insertBefore(par.childNodes[i], par);

			// Remove the parent
			par.parentNode.removeChild(par);
			return true;
		}
		node = par;
		if(i++ > 5)
			break;
	}
	return false;
}

function increase(tag, condemned)
{
	if(remove(condemned))
		return;
	let element = document.createElement(tag);
	let range = window.getSelection().getRangeAt(0);
	range.surroundContents(element);
}

function toggle(tag)
{
	increase(tag, tag);
}

function indent()
{
	let range = window.getSelection().getRangeAt(0);
	let node = range.commonAncestorContainer;
/*
	int i = 0;
	while(node.childNodes.length > 0)
	{
		if(node.childNodes.length > 1 && node.childNodes[1].childNodes.length > node.childNodes[0].childNodes.length)
			node = node.childNodes[1];
		else
			node = node.childNodes[0];
		i++;
		if(i > 2)
			break;
	}
*/
	while(node != null)
	{
		if(node.nodeName === "LI")
		{
			// Determine whether it's an OL or a UL
			let ordered = false;
			let nn = node;
			while(nn != null)
			{
				if(nn.nodeName === "OL")
				{
					ordered = true;
					break;
				}
				nn = nn.parentNode;
			}

			let wrap = document.createElement(ordered ? "OL" : "UL");
			node.parentNode.insertBefore(wrap, node);
			node.parentNode.removeChild(node);
			wrap.appendChild(node);
			return;
		}
		node = node.parentNode;
	}

	increase('MENU');
}

function addLi(ordered)
{
	let wrapName = (ordered ? "OL" : "UL");
	let addwrapper = true;
	let range = window.getSelection().getRangeAt(0);
	let node = range.commonAncestorContainer;
	while(node != null)
	{
		if(node.nodeName === wrapName)
		{
			addwrapper = false;
			break;
		}
		node = node.parentNode;
	}
	let element = document.createElement("LI");
//	let range = window.getSelection().getRangeAt(0);
	range.surroundContents(element);
	if(addwrapper)
		range.surroundContents(document.createElement(wrapName));
}

function addTable()
{
	let table = document.createElement("TABLE");

	// Row #1 (with bold underlined headings)
	let tr = document.createElement("TR");	table.appendChild(tr);
	let td = document.createElement("TD");	tr.appendChild(td);
	let b = document.createElement("B");	td.appendChild(b);
	let u = document.createElement("U");	b.appendChild(u);
	u.appendChild(document.createTextNode("Col 1"));
	td = document.createElement("TD");	tr.appendChild(td);
	b = document.createElement("B");	td.appendChild(b);
	u = document.createElement("U");	b.appendChild(u);
	u.appendChild(document.createTextNode("Col 2"));

	// Row #2
	tr = document.createElement("TR");	table.appendChild(tr);
	td = document.createElement("TD");	tr.appendChild(td);
	td.appendChild(document.createTextNode("El 1"));
	td = document.createElement("TD");	tr.appendChild(td);
	td.appendChild(document.createTextNode("El 2"));

	let range = window.getSelection().getRangeAt(0);
	range.insertNode(table);
}

function removeFormatting()
{
	while(remove("H1")) {}
	while(remove("H2")) {}
	while(remove("H3")) {}
	while(remove("H4")) {}
	while(remove("B")) {}
	while(remove("I")) {}
	while(remove("U")) {}
	while(remove("STRIKE")) {}
	while(remove("SUP")) {}
	while(remove("SUB")) {}
	while(remove("BIG")) {}
	while(remove("SMALL")) {}
	while(remove("CENTER")) {}
}

function setColor(col)
{
	let range = window.getSelection().getRangeAt(0);
	let node = range.commonAncestorContainer;
	while(node.classList === undefined)
		node = node.parentNode;
	node.classList.remove("col_white");
	node.classList.remove("col_gray");
	node.classList.remove("col_black");
	node.classList.remove("col_red");
	node.classList.remove("col_yellow");
	node.classList.remove("col_green");
	node.classList.remove("col_cyan");
	node.classList.remove("col_blue");
	node.classList.remove("col_magenta");
	node.classList.add(col);
}

function setBGCol(col)
{
	let range = window.getSelection().getRangeAt(0);
	let node = range.commonAncestorContainer;
	while(node.classList === undefined)
		node = node.parentNode;
	node.classList.remove("bg_white");
	node.classList.remove("bg_gray");
	node.classList.remove("bg_black");
	node.classList.remove("bg_red");
	node.classList.remove("bg_yellow");
	node.classList.remove("bg_green");
	node.classList.remove("bg_cyan");
	node.classList.remove("bg_blue");
	node.classList.remove("bg_magenta");
	node.classList.add(col);
}

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
	ob.action = "save_gui";
	ob.content = document.getElementById("content").outerHTML;
	ob.filename = document.getElementById("filename").value;
	httpPost("/a", JSON.stringify(ob), cb);
}

/*
let keyStrokes = 0;

function onKey(event)
{
	keyStrokes++;
	if(keyStrokes > 100)
	{
		// save the page
		keyStrokes = 0;
	}
}

document.addEventListener("keypress", onKey);
*/
