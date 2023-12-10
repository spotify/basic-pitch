$(document).ready(function() {
    $('form#transcribe_form').submit(function(event) { 
        event.preventDefault();
        var formData = new FormData(this);
        console.log(formData)

        // call python script to transcribe audio
        server_request('../cgi-bin/transcribe.py', 'POST', formData).then(function(response) {
            if (response.status == 'success') {
                parse_response(response.data)
            } else {
                console.log(response.data)
            }
        });
    });
});

function parse_response(data) {
    console.log(data)
}

function server_request(url, method, data = null) {
    return new Promise(function(resolve) {
        $.ajax({
            type: method,
            url: url,
            data: data,
            processData: false, // Prevent serialization of the FormData object
            contentType: false, // Let the browser set the correct content type for FormData
            success: function (response, status) {
                $("#response").html(response)
                // AJAX Success, server responded
                try {
                    // Try to parse response as JSON
                    response = JSON.parse(response);
                } catch (e) {
                    // Response is not JSON, Error in PHP was not caught
                    response = {
                        status: 'error', 
                        data: "Error parsing response from server.\n\nJS ERROR\n" + e + "\n\nPYTHON ERROR\n" + response
                    };
                    resolve(response);
                }
                // Response successfully parsed as JSON, PHP compiled and caught all errors
                resolve(response);
            },
            error: function (XMLHttpRequest, textStatus, errorThrown) {
                // AJAX Error, server did not respond
                response = {
                    status: 'error', 
                    data: "AJAX ERROR (" + textStatus + ")\n" + errorThrown
                };
                resolve(response);
            }
        })
    });
}