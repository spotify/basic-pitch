$(document).ready(function() {
    $('form#transcribe_form').submit(function(event) { 
        event.preventDefault();
        var formData = new FormData(this);
        
        transcribe_audio(formData);
    });
});

function transcribe_audio(formData) {
    // disable transcribe button and show loading animation
    $('#transcribe_button').prop('disabled', true);
    $("#input_select_model").prop('disabled', true);
    $("#model_results").hide();
    $("#results_error").hide();
    $("#results_error").html('');
    $('#results_loading').show();

    // call python script to transcribe audio
    server_request('../../cgi-bin/transcribe.py', 'POST', formData).then(function(response) {
        switch (response.status) {
            case 'success':
                display_midi(response.data);
                break;
            case 'failure':
                display_error(response.data);
                break;
            case 'error':
                display_error(response.data);
                break;
            default:
                console.log(response)
                break;
        }
    });
}

function display_midi(data) {
    console.log(data)

    $('#transcribe_button').prop('disabled', false);
    $("#input_select_model").prop('disabled', false);
    $("#results_error").hide();
    $("#results_error").html('');
    $('#results_loading').hide();
    
    load_midi("../" + data)
    $("#model_results").show();
}

function display_error(message) {
    $('#transcribe_button').prop('disabled', false);
    $("#input_select_model").prop('disabled', false);
    $("#results_error").html(message);
    $("#results_error").show();
    $('#results_loading').hide();
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
            },
            statusCode: {
                500: function() {
                    alert("Internal server error. Please try again later.");
                 }
              },
        })
    });
}