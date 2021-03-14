function show_result(response) {
	$('.main').addClass('result-mode');
	// "The article might contain false information."
	// "We believe the provided article is true."

	data = JSON.parse(response)
	
	if (data["prediction"] == true) {
		$('.result__icon').html(`<use xlink:href="/static/check-marks.svg#icon-check"></use>`);
		$('.result__icon').css('fill', '#00FF7F')
		$('.result__text').html("We believe the provided article is true.");
		$('.result__prob').html("Probability of True: " + (data["prob"] * 100).toFixed(2) + " %");
	} else {
		$('.result__icon').html(`<use xlink:href="/static/check-marks.svg#icon-cross"></use>`);
		$('.result__icon').css('fill', '#DC143C')
		$('.result__text').html("The article might contain false information.");
		$('.result__prob').html("Probability of True: " + (data["prob"] * 100).toFixed(2) + " %");
	}
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