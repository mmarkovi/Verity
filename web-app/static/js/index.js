function show_result(response) {
	$('.main').addClass('result-mode')
}

$("#article").on( "submit", (event) => {
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
	
	data = JSON.stringify({"title": title, "content": content});

	console.log($(".form").attr("action"));

	$.ajax({
		type: 'GET',
		url: $(".form").attr("action"),
		data: `title=${title}&content=${content}`,
        dataType: 'text',

		success: show_result,
	});
});