$( "#article" ).on( "submit", (event) => {
	event.preventDefault();

	let title = $("#title").val();
	let content = $("#content").val();

	// check if the title is null, empty, or whitespaces only
	if (title.length === 0 || !title.trim()) {
		title = "";
	}

	// check if the content is null, empty, or whitespaces only
	if (content.length === 0 || !content.trim()) {
		content = "";
	}
	
	data = {"title": title, "content": content};

	console.log($(".form").attr("action"))

	$.ajax({
		type: 'POST',
		url: $(".form").attr("action"),
		data: data,
		contentType: 'application/json',
        dataType: 'json',

		success: (response) => {alert(response)},
	});
});

$.ajax({
	type: 'POST',
	url: $(".form").attr("action"),
	data: $(".form").serialize(), 
	//or your custom data either as object {foo: "bar", ...} or foo=bar&...
	success: function(response) {},
  });